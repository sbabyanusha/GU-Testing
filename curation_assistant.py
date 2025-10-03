from __future__ import annotations
"""
Universal Figure Interpreter (Streamlit)
---------------------------------------
- Ingest PDFs/DOCX/TXT/CSV/XLSX into a lightweight local RAG index
- Summarize and Q&A strictly from uploaded documents
- Extract & analyze figures from PDFs or uploaded images (OCR, sub-panels, chart guess, scale bar, colors, EXIF)
- Auto-classify ANY image (histology, plot, table, pipeline/diagram, UI, photo) and generate a grounded interpretation
- No style choices; always selects the best voice automatically
"""
import sys, io, re, tempfile, textwrap, datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
import streamlit as st, pandas as pd
from PIL import Image, ExifTags
try:
    import pysqlite3; sys.modules["sqlite3"]=sys.modules["pysqlite3"]
except Exception: pass
try:
    import numpy as np
    if not hasattr(np,"float_"): np.float_ = np.float64
except Exception:
    import numpy as np  # type: ignore
# LangChain
Chroma=None; CLIENT=None
try:
    from langchain_chroma import Chroma as _Chroma; Chroma=_Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma as _Chroma; Chroma=_Chroma
    except Exception: pass
try:
    from chromadb import PersistentClient; CLIENT=PersistentClient(path="./chroma_index")
except Exception: CLIENT=None
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import docx, mammoth
# Vision
try:
    import cv2; CV2_OK=True
except Exception as e:
    CV2_OK=False
from skimage.util import img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
try:
    import pytesseract; TESS_AVAILABLE=True
except Exception:
    TESS_AVAILABLE=False

st.set_page_config(page_title="🧠 Universal Curation Tool", layout="wide")
CHUNK_SIZE=1200; CHUNK_OVERLAP=150; TOP_K=5; MMR_LAMBDA=0.5
COLLECTION="curation_assistant"; EMBED_MODEL="text-embedding-3-small"; LLM_MODEL="gpt-4o-mini"
for k,v in [("vectorstore",None),("all_docs",[]),("all_figs",[]),("backend",None)]:
    if k not in st.session_state: st.session_state[k]=v

def _read_txt(file: io.BytesIO, name: str) -> List[Document]:
    text=file.read().decode("utf-8",errors="ignore")
    return [Document(page_content=text, metadata={"source":name})]

def _read_pdf_to_docs(tmp: Path, name: str) -> List[Document]:
    loader=PyPDFLoader(str(tmp)); docs=loader.load()
    for d in docs: d.metadata={**d.metadata, "source":name}
    return docs

def _extract_pdf_images(tmp: Path, name: str):
    from pypdf import PdfReader
    pil_images=[]; ocr_docs=[]
    try:
        reader=PdfReader(str(tmp))
        for i,page in enumerate(reader.pages, start=1):
            try: page_text=page.extract_text() or ""
            except Exception: page_text=""
            for im in getattr(page, "images", []):
                try:
                    img=Image.open(io.BytesIO(im.data)).convert("RGB"); pil_images.append(img)
                    ocr_text=""
                    if TESS_AVAILABLE:
                        try: ocr_text=pytesseract.image_to_string(img)
                        except Exception: pass
                    cap="\n".join([ln for ln in page_text.splitlines() if re.match(r"\s*Fig(ure)?\b", ln, re.I)])
                    combo="\n".join([s for s in [cap.strip(), (ocr_text or '').strip()] if s])
                    if combo:
                        ocr_docs.append(Document(page_content=f"[FIGURE OCR p.{i}]\n{combo}",
                                                 metadata={"source":name,"page":i,"type":"figure_ocr"}))
                except Exception: continue
    except Exception: pass
    return ocr_docs, pil_images

def _read_docx(file: io.BytesIO, name: str) -> List[Document]:
    try:
        d=docx.Document(io.BytesIO(file.read()))
        text="\n".join([p.text for p in d.paragraphs if p.text.strip()])
        return [Document(page_content=text, metadata={"source":name})]
    except Exception:
        file.seek(0); text=mammoth.extract_raw_text(file).value or ""
        return [Document(page_content=text, metadata={"source":name})]

def _read_xlsx(file: io.BytesIO, name: str) -> List[Document]:
    docs=[]
    try:
        xl=pd.ExcelFile(file)
        for sheet in xl.sheet_names:
            df=xl.parse(sheet).astype(str)
            lines=["\t".join(df.columns.astype(str))]+["\t".join(map(str,r)) for r in df.itertuples(index=False, name=None)]
            docs.append(Document(page_content="\n".join(lines), metadata={"source":name,"sheet":sheet}))
    except Exception as e:
        st.error(f"Excel parse failed: {e}")
    return docs

def load_files_to_documents(files):
    all_docs=[]; all_figs=[]
    for uf in files or []:
        suf=Path(uf.name).suffix.lower()
        if suf in [".txt",".md",".csv"]: all_docs+=_read_txt(uf, uf.name)
        elif suf==".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                tmp.write(uf.read()); tmp.flush()
                all_docs+=_read_pdf_to_docs(Path(tmp.name), uf.name)
                ocr,figs=_extract_pdf_images(Path(tmp.name), uf.name)
                all_docs+=ocr; all_figs+=figs
        elif suf==".docx": all_docs+=_read_docx(uf, uf.name)
        elif suf in [".xlsx",".xlsm",".xls"]: all_docs+=_read_xlsx(uf, uf.name)
        else: st.warning(f"Unsupported file: {uf.name}")
    return all_docs, all_figs

def _split(docs):
    s=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    parts=s.split_documents(docs)
    if not parts: raise ValueError("No text extracted.")
    return parts

def build_index(docs):
    splits=_split(docs); emb=OpenAIEmbeddings(model=EMBED_MODEL)
    if Chroma and CLIENT:
        try:
            vs=Chroma.from_documents(splits, emb, client=CLIENT, collection_name=COLLECTION)
            try: vs.persist()
            except Exception: pass
            return vs, emb, splits, "chroma"
        except Exception as e:
            st.sidebar.error(f"Chroma failed; using FAISS: {e}")
    return FAISS.from_documents(splits, emb), emb, splits, "faiss"

def load_existing_index():
    emb=OpenAIEmbeddings(model=EMBED_MODEL)
    if Chroma and CLIENT:
        try: return Chroma(client=CLIENT, collection_name=COLLECTION, embedding_function=emb), emb, "chroma"
        except Exception as e: st.sidebar.error(f"No existing collection: {e}")
    return None, emb, "faiss"

SUMMARY_SYSTEM="You are a meticulous scientific analyst. Summarize strictly from the provided content."
SUMMARY_USER="Summarize the following corpus in <= 20 bullet points. Group by source when helpful.\n\n{context}"
summary_prompt=ChatPromptTemplate.from_messages([("system",SUMMARY_SYSTEM),("human",SUMMARY_USER)])
QA_SYSTEM="Answer ONLY using the given context. If not in context, say 'I don't know from the provided files.'"
QA_USER="Question: {question}\n\nUse the context to answer concisely in 1-15 sentences.\n\nContext:\n{context}"
qa_prompt=ChatPromptTemplate.from_messages([("system",QA_SYSTEM),("human",QA_USER)])

def to_cv(img: Image.Image):
    if not CV2_OK: return None
    arr=np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def read_exif(img: Image.Image) -> Dict[str,Any]:
    meta={}
    try:
        exif=img.getexif()
        if exif:
            for tid,val in exif.items():
                tag=ExifTags.TAGS.get(tid, tid); meta[str(tag)]=str(val)
    except Exception: pass
    return meta

def ocr_image(img: Image.Image, lang="eng") -> str:
    if not TESS_AVAILABLE: return ""
    try: return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception: return ""

def _axes_stats(gray_u8):
    edges=canny(gray_u8/255.0, sigma=1.5)
    h,theta,d=hough_line(edges); _,angles,_=hough_line_peaks(h,theta,d,num_peaks=20)
    vert=sum(1 for a in angles if abs(np.degrees(a))<10 or abs(np.degrees(a)-180)<10)
    horiz=sum(1 for a in angles if abs(abs(np.degrees(a))-90)<10)
    return {"num_lines":int(len(angles)),"vertical":int(vert),"horizontal":int(horiz),"edge_density":float(np.mean(edges))}

def infer_chart_type(gray_u8):
    ax=_axes_stats(gray_u8)
    if ax["horizontal"]>=1 and ax["vertical"]>=1:
        return "line/scatter (axes detected)" if ax["edge_density"]>0.08 else "bar-like (axes detected)"
    if ax["num_lines"]>=6: return "grid/heatmap-like (many straight lines)"
    return "illustration/microscopy/other"

def detect_circos_ring(cv_bgr):
    if not CV2_OK or cv_bgr is None: return 0
    gray=cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=80, param1=80, param2=30, minRadius=80, maxRadius=0)
    return 0 if circles is None else circles.shape[1]

def detect_leader_lines_and_callouts(cv_bgr):
    if not CV2_OK or cv_bgr is None: return (0,0)
    gray=cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray, 50, 150, apertureSize=3)
    lines=cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=max(60, gray.shape[1]//5), maxLineGap=15)
    long_horiz=0
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1-y2)<5 and abs(x2-x1)>gray.shape[1]//4:
                long_horiz+=1
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, param1=80, param2=30, minRadius=6, maxRadius=30)
    callouts=0 if circles is None else circles.shape[1]
    return long_horiz, callouts

def dominant_colors(cv_bgr,k=5):
    if not CV2_OK or cv_bgr is None: return []
    data=cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    _,labels,centers=cv2.kmeans(data,k,None,criteria,3,cv2.KMEANS_PP_CENTERS)
    centers=centers.astype(int); counts=np.bincount(labels.flatten()); order=np.argsort(-counts)
    return [tuple(map(int, centers[i])) for i in order]

def analyze_image(img: Image.Image, lang="eng"):
    if not CV2_OK:
        return {"width":img.width,"height":img.height,"ocr_text":ocr_image(img,lang),
                "chart_guess":"(cv2 unavailable)","dominant_colors":[], "exif":read_exif(img),
                "edge_density":0.0,"anatomy_cues":(0,0),"circos_circles":0}
    cv_img=to_cv(img); gray=cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray_u8=img_as_ubyte(gray/255.0)
    ocr=ocr_image(img, lang=lang)
    ax=_axes_stats(gray_u8)
    return {"width":img.width,"height":img.height,"ocr_text":ocr,"chart_guess":infer_chart_type(gray_u8),
            "dominant_colors":dominant_colors(cv_img,5),"exif":read_exif(img),
            "edge_density":ax["edge_density"],"anatomy_cues":detect_leader_lines_and_callouts(cv_img),
            "circos_circles":detect_circos_ring(cv_img)}

def _text_keywords(s: str):
    s=(s or "").lower()
    return {
        "genome_words": any(k in s for k in ["chr1","chr2","chrx","chry","ideogram","loh","copy number","cnv","snv","mutation","indel","structural variant","translocation","inversion","duplication","deletion","circos"]),
        "mutation_key": any(k in s for k in ["c>a","c>g","c>t","t>a","t>c","t>g"]),
        "he_words": any(k in s for k in ["h&e","hematoxylin","eosin"]),
        "anatomy_words": any(k in s for k in ["anatomy","liver","lung","kidney","pancreas","intestine","stomach","brain","heart"]),
        "pipeline_words": any(k in s for k in ["pipeline","workflow","etl","deploy","module","transfer","import"]),
        "ui_words": any(k in s for k in ["button","login","settings","window","screenshot"]),
        "table_words": any(k in s for k in ["table","row","column","supplementary"]),
    }

def _palette_hint(colors):
    def _name(c):
        r,g,b=c
        if r>180 and g<160 and b<160: return "pink"
        if r>150 and b>150 and g<140: return "purple"
        if r<80 and g<80 and b<80:   return "dark"
        return "other"
    names=[_name(c) for c in (colors or [])[:6]]
    return {"he_like": (names.count("pink")+names.count("purple"))>=2}

def classify_image_any(rep: dict):
    reasons=[]; ocr=(rep.get("ocr_text") or "")
    kw=_text_keywords(ocr); pal=_palette_hint(rep.get("dominant_colors", []))
    cg=(rep.get("chart_guess") or "").lower()
    long_horiz, callouts = rep.get("anatomy_cues",(0,0))
    circos = int(rep.get("circos_circles",0))
    axes_like=("axes detected" in cg); grid_like=("grid" in cg)
    votes={"genomics_plot":0,"anatomy":0,"histology":0,"chart":0,"table":0,"pipeline":0,"ui":0,"photo":0}
    if kw["genome_words"]: votes["genomics_plot"]+=3; reasons.append("genome OCR terms")
    if kw["mutation_key"]: votes["genomics_plot"]+=1; reasons.append("SNV legend")
    if circos>=1: votes["genomics_plot"]+=2; reasons.append("circos rings")
    if long_horiz>=2: votes["anatomy"]+=2; reasons.append(f"{long_horiz} leader lines")
    if callouts>=2: votes["anatomy"]+=2; reasons.append(f"{callouts} callouts")
    if kw["anatomy_words"]: votes["anatomy"]+=1
    if pal["he_like"] or kw["he_words"]: votes["histology"]+=1; reasons.append("H&E-like cues")
    if axes_like: votes["chart"]+=1; reasons.append("axes")
    if grid_like: votes["table"]+=1; reasons.append("grid-like")
    if kw["pipeline_words"]: votes["pipeline"]+=2; reasons.append("pipeline words")
    if kw["ui_words"]: votes["ui"]+=2; reasons.append("UI words")
    if kw["table_words"]: votes["table"]+=1
    if not any(votes.values()): votes["photo"]+=1; reasons.append("no structural cues")
    figure_type=max(votes, key=votes.get); conf=min(1.0, 0.35 + 0.18*votes[figure_type])
    if votes[figure_type]==0: figure_type, conf="unknown", 0.3
    return figure_type, conf, reasons

def _brief_palette_desc(colors):
    if not colors: return "no palette extracted"
    def _name(c):
        r,g,b=c
        if r>200 and g>200 and b>200: return "white"
        if r>150 and b>150 and g<140: return "lavender/purple"
        if r>180 and g<160 and b<160: return "pink"
        if r<80 and g<80 and b<80:   return "dark/gray"
        return "mixed"
    names=[_name(c) for c in colors[:5]]
    out=[]
    for n in names:
        if not out or out[-1]!=n: out.append(n)
    return ", ".join(out)

def _type_hint(fig_type: str) -> str:
    return {
        "genomics_plot":"Explain chromosomes/ideogram, SNVs/indels, copy-number gains/losses, LOH, and structural variant arcs. Do not invent gene names or values.",
        "anatomy":"Describe body regions indicated by leader lines/callouts without inventing labels.",
        "histology":"Describe tissue architecture and stain patterns; do not guess diagnosis.",
        "chart":"Describe axes, groups, and visible trends; avoid numbers unless in OCR.",
        "table":"Describe rows/columns/headings inferred by OCR; no invented values.",
        "pipeline":"Summarize modules and flow (inputs→processing→outputs).",
        "ui":"Summarize layout and controls.",
        "photo":"Describe salient objects and relationships.",
        "unknown":"Be cautious; stick to cues."
    }.get(fig_type,"Be cautious; stick to cues.")

def _build_context(rep: dict, fig_type: str) -> str:
    pal=_brief_palette_desc(rep.get("dominant_colors", []))
    ocr=(rep.get("ocr_text") or "").strip()
    ocr_short=textwrap.shorten(ocr.replace("\n"," "), width=600, placeholder="…") if ocr else ""
    long_h, callouts = rep.get("anatomy_cues",(0,0))
    return f"FIGURE_TYPE: {fig_type}\nWIDTH: {rep.get('width')}  HEIGHT: {rep.get('height')}\nCHART_GUESS: {rep.get('chart_guess','')}\nCIRCOS_RINGS_DETECTED: {rep.get('circos_circles',0)}\nANATOMY_CUES: leader_lines={long_h}, callouts={callouts}\nPALETTE: {pal}\nOCR_SNIPPET:\n{ocr_short or '(none)'}"

def generate_interpretation_md(rep: dict, llm_model="gpt-4o-mini") -> str:
    fig_type, conf, reasons = classify_image_any(rep)
    context = _build_context(rep, fig_type)
    try:
        llm=ChatOpenAI(model=llm_model, temperature=0.15, max_tokens=900)
    except Exception:
        return f"**Heuristic** — {fig_type} (conf≈{conf:.2f}). Cues: {', '.join(reasons)}\n\n{context}"
    system=("You are an expert figure interpreter across domains. Use ONLY the provided context/OCR/cues; never invent labels or numbers. "
            "If info is missing, say so.")
    user=f"""CLASSIFIER_CONFIDENCE: {conf:.2f}
CLASSIFIER_REASONS: {', '.join(reasons) if reasons else '(none)'}
TYPE_HINT: {_type_hint(fig_type)}

CONTEXT:
{context}

Return three sections in Markdown:
1) **Interpretation** (4–8 sentences, type-specific)
2) **Evidence mapping** (bullets citing concrete cues)
3) **Caption/Legend (draft)** (3–6 sentences)
"""
    prompt=ChatPromptTemplate.from_messages([("system",system),("human",user)])
    chain=prompt|llm|StrOutputParser()
    try: return chain.invoke({})
    except Exception: return f"**Heuristic** — {fig_type} (conf≈{conf:.2f}). Cues: {', '.join(reasons)}\n\n{context}"

# ---- Sidebar ----
with st.sidebar:
    st.subheader("Upload files")
    uploads=st.file_uploader("Drop PDF/TXT/DOC/DOCX/XLSX files",
                             type=["pdf","txt","md","csv","doc","docx","xlsx","xlsm","xls"],
                             accept_multiple_files=True)
    c1,c2=st.columns(2)
    with c1: build_btn=st.button("Ingest Files", type="primary")
    with c2: load_btn=st.button("Load Existing Index")

if build_btn:
    if not uploads: st.warning("Please upload at least one file.")
    else:
        with st.spinner("Parsing & embedding…"):
            docs, figs = load_files_to_documents(uploads)
            if docs:
                vs,_,splits,backend=build_index(docs)
                st.session_state.vectorstore=vs; st.session_state.all_docs=splits; st.session_state.all_figs=figs; st.session_state.backend=backend
                st.success(f"Indexed {len(splits)} chunks ({backend}). Figures: {len(figs)}.")
            else:
                st.error("No parsable content found.")

if load_btn:
    with st.spinner("Loading existing index…"):
        vs,_,backend=load_existing_index()
        if vs is not None:
            st.session_state.vectorstore=vs; st.session_state.backend=backend
            st.info(f"Loaded existing collection with backend: {backend}.")
        else:
            st.warning("No persisted index found; please ingest files.")

summary_tab, qa_tab, figs_tab = st.tabs(["📝 Summary","❓ Q&A","🖼️ Figures"])

with summary_tab:
    if st.session_state.vectorstore is None and not st.session_state.all_docs:
        st.info("Build or load an index to enable summarization.")
    else:
        if st.button("Generate Summary"):
            subset=list(st.session_state.all_docs)[:25]
            if subset:
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
                chain=summary_prompt|llm|StrOutputParser()
                ctx="\n\n".join([f"[src={d.metadata.get('source','-')} sheet={d.metadata.get('sheet','-')} page={d.metadata.get('page','-')}]\n{d.page_content}" for d in subset])
                st.markdown(chain.invoke({"context":ctx}))
            else:
                st.warning("No chunks to summarize.")

with qa_tab:
    if st.session_state.vectorstore is None:
        st.info("Build or load an index to enable Q&A.")
    else:
        q=st.text_input("Your question")
        if st.button("Ask", type="primary") and q.strip():
            retr=st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":TOP_K,"lambda_mult":MMR_LAMBDA})
            docs=retr.get_relevant_documents(q)
            if not docs:
                st.warning("No sufficiently relevant context found. I don't know from the provided files.")
            else:
                ctx="\n\n".join([f"[src={d.metadata.get('source','-')} sheet={d.metadata.get('sheet','-')} page={d.metadata.get('page','-')}]\n{d.page_content}" for d in docs])
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
                chain=qa_prompt|llm|StrOutputParser()
                st.markdown("**Answer**")
                st.write(chain.invoke({"question":q,"context":ctx}))

with figs_tab:
    st.write("Upload images below or rely on figures extracted from PDFs during ingestion.")
    lang=st.text_input("OCR language", value="eng")
    imgs=st.file_uploader("Upload images (PNG/JPG/TIFF)", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True)
    figures=list(st.session_state.all_figs)
    if imgs:
        for f in imgs:
            try: figures.append(Image.open(f).convert("RGB"))
            except Exception: st.warning(f"Could not open {f.name}")
    if not figures:
        st.info("No figures yet — upload images above.")
    else:
        for i,img in enumerate(figures, start=1):
            st.markdown(f"### Figure {i}")
            st.image(img, use_container_width=True)
            rep=analyze_image(img, lang=lang)
            c1,c2,c3=st.columns(3)
            with c1:
                st.subheader("Chart guess"); st.write(rep["chart_guess"])
                st.caption(f"Circos rings: {rep.get('circos_circles',0)} | Leader lines: {rep.get('anatomy_cues',(0,0))[0]}")
            with c2:
                st.subheader("Top colors")
                cols=rep["dominant_colors"]
                if cols:
                    import numpy as np, PIL.Image as PILI
                    arr=np.zeros((30,50*len(cols),3), dtype=np.uint8)
                    for j,(r,g,b) in enumerate(cols): arr[:, j*50:(j+1)*50,:]=[r,g,b]
                    st.image(PILI.fromarray(arr), use_container_width=True)
                else:
                    st.write("(no palette)")
            with c3:
                st.subheader("EXIF / Metadata"); st.json(rep["exif"] or {"note":"no EXIF"})
            with st.expander("OCR (full image)"):
                st.code(rep["ocr_text"] or "(no OCR text)")
            if st.button("Generate interpretation", key=f"interp_{i}"):
                md=generate_interpretation_md(rep)
                st.markdown(md)
                ts=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                st.download_button("⬇️ Download interpretation (Markdown)",
                    data=md.encode("utf-8"), file_name=f"figure_{i}_interpretation_{ts}.md", mime="text/markdown", use_container_width=True)
            st.markdown("---")
