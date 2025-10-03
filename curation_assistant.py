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

# ===== Compatibility shims =====
import sys, io, re, json, tempfile, textwrap, datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except Exception:
    pass

try:
    import numpy as np
    if not hasattr(np, "float_"):
        np.float_ = np.float64
except Exception:
    import numpy as np  # type: ignore

# ===== Core libs =====
import streamlit as st
import pandas as pd
from PIL import Image, ExifTags

# LangChain pieces
Chroma = None
CLIENT = None
try:
    from langchain_chroma import Chroma as _Chroma
    Chroma = _Chroma
except Exception:
    try:
        from langchain_community.vectorstores import Chroma as _Chroma
        Chroma = _Chroma
    except Exception:
        pass

try:
    from chromadb import PersistentClient
    CLIENT = PersistentClient(path="./chroma_index")
except Exception:
    CLIENT = None

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import docx
import mammoth

# Vision deps
try:
    import cv2
    CV2_OK = True
    CV2_ERR = None
except Exception as e:
    CV2_OK = False
    CV2_ERR = e

from skimage.util import img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

try:
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

# ===== UI setup =====
st.set_page_config(page_title="🧬 Universal Curation Tool", layout="wide")

if not CV2_OK:
    st.warning(
        "OpenCV (cv2) is not available — figure analysis limited. "
        "Install `opencv-python-headless==4.10.0.84` and redeploy. "
        f"(Import error: {CV2_ERR})"
    )

# ===== Constants & session =====
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 5
MMR_LAMBDA = 0.5
COLLECTION = "curation_assistant"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

for k, v in [("vectorstore", None), ("all_docs", []), ("all_figs", []), ("backend", None)]:
    if k not in st.session_state: st.session_state[k] = v

# ===== Readers =====
def _read_txt(file: io.BytesIO, name: str) -> List[Document]:
    text = file.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": name})]

def _read_pdf_to_docs(tmp_path: Path, name: str) -> List[Document]:
    loader = PyPDFLoader(str(tmp_path)); docs = loader.load()
    for d in docs: d.metadata = {**d.metadata, "source": name}
    return docs

def _extract_pdf_images(tmp_path: Path, name: str) -> Tuple[List[Document], List[Image.Image]]:
    from pypdf import PdfReader
    pil_images: List[Image.Image] = []; ocr_docs: List[Document] = []
    try:
        reader = PdfReader(str(tmp_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            try: page_text = page.extract_text() or ""
            except Exception: page_text = ""
            for im in getattr(page, "images", []):
                try:
                    img = Image.open(io.BytesIO(im.data)).convert("RGB")
                    pil_images.append(img)
                    ocr_text = ""
                    if TESS_AVAILABLE:
                        try: ocr_text = pytesseract.image_to_string(img)
                        except Exception: pass
                    cap = "\n".join([ln for ln in page_text.splitlines() if re.match(r"\s*Fig(ure)?\b", ln, re.I)])
                    combined = "\n".join([s for s in [cap.strip(), (ocr_text or "").strip()] if s])
                    if combined:
                        ocr_docs.append(Document(page_content=f"[FIGURE OCR p.{page_idx}]\n{combined}",
                                                 metadata={"source": name, "page": page_idx, "type": "figure_ocr"}))
                except Exception:
                    continue
    except Exception:
        pass
    return ocr_docs, pil_images

def _read_docx(file: io.BytesIO, name: str) -> List[Document]:
    try:
        f = io.BytesIO(file.read()); d = docx.Document(f)
        text = "\n".join([p.text for p in d.paragraphs if p.text.strip()])
        return [Document(page_content=text, metadata={"source": name})]
    except Exception:
        file.seek(0); text = mammoth.extract_raw_text(file).value or ""
        return [Document(page_content=text, metadata={"source": name})]

def _read_xlsx(file: io.BytesIO, name: str) -> List[Document]:
    docs: List[Document] = []
    try:
        xl = pd.ExcelFile(file)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet).astype(str)
            lines = ["\t".join(df.columns.astype(str))] + ["\t".join(map(str, r)) for r in df.itertuples(index=False, name=None)]
            docs.append(Document(page_content="\n".join(lines), metadata={"source": name, "sheet": sheet}))
        return docs
    except Exception as e:
        st.error(f"Excel parse failed for {name}: {e}"); return []

def load_files_to_documents(uploaded_files) -> Tuple[List[Document], List[Image.Image]]:
    all_docs: List[Document] = []; all_figs: List[Image.Image] = []
    for uf in uploaded_files:
        suf = Path(uf.name).suffix.lower()
        if suf in [".txt",".md",".csv"]: all_docs.extend(_read_txt(uf, uf.name))
        elif suf == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                tmp.write(uf.read()); tmp.flush()
                all_docs.extend(_read_pdf_to_docs(Path(tmp.name), uf.name))
                ocr_docs, figs = _extract_pdf_images(Path(tmp.name), uf.name)
                all_docs.extend(ocr_docs); all_figs.extend(figs)
        elif suf == ".docx": all_docs.extend(_read_docx(uf, uf.name))
        elif suf in [".xlsx",".xlsm",".xls"]: all_docs.extend(_read_xlsx(uf, uf.name))
        else: st.warning(f"Unsupported file: {uf.name}")
    return all_docs, all_figs

# ===== Index =====
def _split_docs(docs: List[Document]):
    s = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    parts = s.split_documents(docs)
    if not parts: raise ValueError("No text extracted from uploads.")
    return parts

def build_index(docs: List[Document]):
    splits = _split_docs(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if Chroma is not None and CLIENT is not None:
        try:
            vs = Chroma.from_documents(splits, embeddings, client=CLIENT, collection_name=COLLECTION)
            try: vs.persist()
            except Exception: pass
            return vs, embeddings, splits, "chroma"
        except Exception as e:
            st.sidebar.error(f"Chroma failed; using FAISS: {e}")
    vs = FAISS.from_documents(splits, embeddings)
    return vs, embeddings, splits, "faiss"

def load_existing_index():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if Chroma is not None and CLIENT is not None:
        try: return Chroma(client=CLIENT, collection_name=COLLECTION, embedding_function=embeddings), embeddings, "chroma"
        except Exception as e: st.sidebar.error(f"No existing Chroma collection: {e}")
    return None, embeddings, "faiss"

# ===== Prompts =====
SUMMARY_SYSTEM = (
    "You are a meticulous scientific analyst. Summarize strictly from the provided content; no external knowledge."
)
SUMMARY_USER = "Summarize the following corpus in <= 20 bullet points. Group by source when helpful.\n\n{context}"
summary_prompt = ChatPromptTemplate.from_messages([("system", SUMMARY_SYSTEM), ("human", SUMMARY_USER)])

QA_SYSTEM = (
    "Answer ONLY using the given context. If the answer isn't in the context, say 'I don't know from the provided files.'"
)
QA_USER = "Question: {question}\n\nUse the context to answer concisely in 1-15 sentences.\n\nContext:\n{context}"
qa_prompt = ChatPromptTemplate.from_messages([("system", QA_SYSTEM), ("human", QA_USER)])

# ===== Vision helpers =====
def to_cv(img: Image.Image):
    if not CV2_OK: return None
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def read_exif(img: Image.Image) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        exif = img.getexif()
        if exif:
            for tid, val in exif.items():
                tag = ExifTags.TAGS.get(tid, tid)
                meta[str(tag)] = str(val)
    except Exception: pass
    return meta

def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    if not TESS_AVAILABLE: return ""
    try: return pytesseract.image_to_string(img, lang=lang).strip()
    except Exception: return ""

def detect_subpanels(cv_bgr):
    if not CV2_OK or cv_bgr is None: return []
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H,W = gray.shape[:2]; area_img = W*H
    boxes=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt); area=w*h; aspect=w/(h+1e-6)
        if area>0.02*area_img and 0.2<aspect<5 and w>60 and h>60: boxes.append((x,y,w,h))
    # drop nested
    keep=[]
    for b in boxes:
        x,y,w,h=b; ok=True
        for b2 in boxes:
            if b is b2: continue
            x2,y2,w2,h2=b2
            if x>=x2 and y>=y2 and x+w<=x2+w2 and y+h<=y2+h2 and (w*h)<(w2*h2): ok=False; break
        if ok: keep.append(b)
    keep.sort(key=lambda r:(r[1]//50,r[0]))
    return keep[:12]

def _axes_stats(gray_u8: np.ndarray) -> Dict[str,int]:
    edges = canny(gray_u8/255.0, sigma=1.5)
    h,theta,d = hough_line(edges); _,angles,_ = hough_line_peaks(h,theta,d,num_peaks=15)
    vert = sum(1 for a in angles if abs(np.degrees(a))<10 or abs(np.degrees(a)-180)<10)
    horiz= sum(1 for a in angles if abs(abs(np.degrees(a))-90)<10)
    return {"num_lines":int(len(angles)),"vertical":int(vert),"horizontal":int(horiz)}

def infer_chart_type(gray_u8: np.ndarray) -> str:
    ax = _axes_stats(gray_u8)
    if ax["horizontal"]>=1 and ax["vertical"]>=1:
        edge_density = float(np.mean(canny(gray_u8/255.0)))
        return "line/scatter (axes detected)" if edge_density>0.08 else "bar-like (axes detected)"
    if ax["num_lines"]>=6: return "grid/heatmap-like (many straight lines)"
    return "illustration/microscopy/other"

def detect_scale_bar(cv_bgr, ocr_text: str) -> bool:
    if not CV2_OK or cv_bgr is None: return False
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,10)
    contours,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H,W = gray.shape[:2]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt); aspect=w/(h+1e-6); area=w*h
        if area>0.001*W*H and (aspect>8 or (1/aspect)>8):
            if y>0.7*H or x>0.7*W or x<0.1*W: return True
    low=(ocr_text or "").lower()
    return any(k in low for k in ["scale","µm","um","nm","mm bar","100 µm","50 µm"])

def dominant_colors(cv_bgr,k=5):
    if not CV2_OK or cv_bgr is None: return []
    data = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    _,labels,centers = cv2.kmeans(data,k,None,criteria,3,cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(int); counts = np.bincount(labels.flatten()); order = np.argsort(-counts)
    return [tuple(map(int,centers[i])) for i in order]

def analyze_image(img: Image.Image, lang: str = "eng") -> Dict[str, Any]:
    if not CV2_OK:
        return {"width": img.width, "height": img.height, "ocr_text": ocr_image(img, lang=lang),
                "chart_guess":"(cv2 unavailable)", "has_scale_bar": False,
                "dominant_colors": [], "panels": [], "exif": read_exif(img)}
    cv_img = to_cv(img); gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray_u8 = img_as_ubyte(gray/255.0)
    ocr = ocr_image(img, lang=lang)
    boxes = detect_subpanels(cv_img)
    panels=[]; 
    for (x,y,w,h) in boxes:
        crop = cv_img[y:y+h,x:x+w]
        cg = infer_chart_type(img_as_ubyte(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)/255.0))
        excerpt = ocr_image(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), lang=lang)[:300]
        panels.append({"bbox":(x,y,w,h),"detected_chart":cg,"ocr_excerpt":excerpt})
    return {
        "width": img.width, "height": img.height, "ocr_text": ocr,
        "chart_guess": infer_chart_type(gray_u8),
        "has_scale_bar": detect_scale_bar(cv_img, ocr),
        "dominant_colors": dominant_colors(cv_img, k=5),
        "panels": panels, "exif": read_exif(img)
    }

def draw_boxes(cv_bgr, boxes):
    if not CV2_OK or cv_bgr is None: return None
    out = cv_bgr.copy()
    for i,(x,y,w,h) in enumerate(boxes):
        cv2.rectangle(out,(x,y),(x+w,y+h),(36,255,12),2)
        cv2.putText(out, f"{chr(65 + (i%26))}", (x+5,y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return out

def palette_image(colors):
    if not colors: return Image.new("RGB",(1,1),(255,255,255))
    w=50*len(colors); h=30; img = Image.new("RGB",(w,h),(255,255,255)); arr=np.array(img)
    for i,(r,g,b) in enumerate(colors): arr[:, i*50:(i+1)*50, :] = (int(r),int(g),int(b))
    return Image.fromarray(arr)

# ===== Universal classifier =====
def _text_keywords(s: str):
    s = (s or "").lower()
    return {
        "has_units": any(k in s for k in ["mm","µm","um","nm","°c","%","p-value","auc","r²","r2"]),
        "has_axes_words": any(k in s for k in ["x-axis","y-axis","time","days","weeks"]),
        "table_words": any(k in s for k in ["row","column","table","supplementary"]),
        "pipeline_words": any(k in s for k in ["import_","clone","deploy","etl","pipeline","workflow"]),
        "bio_words": any(k in s for k in ["h&e","hematoxylin","eosin","histology","tumor","gland"]),
        "ui_words": any(k in s for k in ["button","login","settings","screenshot","window"]),
    }

def _palette_hint(colors):
    def _name(c):
        r,g,b=c
        if r>180 and g<160 and b<160: return "pink"
        if r>150 and b>150 and g<140: return "purple"
        if r<80 and g<80 and b<80:   return "dark"
        return "other"
    names=[_name(c) for c in (colors or [])[:6]]
    return {"he_like": (names.count("pink")+names.count("purple"))>=2, "dark_ratio": names.count("dark")/max(1,len(names))}

def classify_image_any(rep: dict) -> tuple[str, float, list[str]]:
    reasons=[]; ocr=(rep.get("ocr_text") or "").strip(); kw=_text_keywords(ocr); pal=_palette_hint(rep.get("dominant_colors", []))
    cg=(rep.get("chart_guess") or "").lower()
    axes_like=("axes detected" in cg) or ("line/scatter" in cg) or ("bar-like" in cg)
    grid_like=("grid" in cg) or ("heatmap" in cg)

    votes={"histology":0,"plot":0,"table":0,"pipeline":0,"ui":0,"photo":0}
    if rep.get("has_scale_bar"): votes["histology"]+=1; reasons.append("scale bar")
    if pal["he_like"]: votes["histology"]+=1; reasons.append("H&E-like palette")
    if axes_like or kw["has_units"] or kw["has_axes_words"]: votes["plot"]+=1; reasons.append("axes/units cues")
    if grid_like: votes["table"]+=1; reasons.append("grid lines")
    if kw["pipeline_words"]: votes["pipeline"]+=2; reasons.append("pipeline keywords")
    if kw["table_words"]: votes["table"]+=1; reasons.append("table keywords")
    if kw["ui_words"]: votes["ui"]+=2; reasons.append("UI keywords")
    if not any(votes.values()) and not axes_like and not grid_like: votes["photo"]+=1; reasons.append("no strong structural cues")

    figure_type = max(votes, key=votes.get)
    conf = min(1.0, 0.4 + 0.2*votes[figure_type])
    if votes[figure_type]==0: figure_type, conf = "unknown", 0.3
    return figure_type, conf, reasons

# ===== Interpretation helpers =====
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
    out=[]; 
    for n in names:
        if not out or out[-1]!=n: out.append(n)
    return ", ".join(out)

def _rule_based_interpretation(rep: dict, ftype: str, conf: float, reasons: list[str]) -> str:
    pal=_brief_palette_desc(rep.get("dominant_colors", []))
    ocr=(rep.get("ocr_text") or "").strip()
    ocr_snip=textwrap.shorten(ocr.replace("\n"," "), width=180, placeholder="…") if ocr else "no OCR text"
    return textwrap.dedent(f"""
    **Interpretation (heuristic):**
    Type guess: {ftype} (conf~{conf:.2f}). Based on cues: {', '.join(reasons) or 'none'}.
    The image contains multiple regions of interest and clear structural cues. Use the evidence below to guide domain-specific reading.

    **Evidence mapping:**
    - Scale bar: {"Yes" if rep.get("has_scale_bar") else "No"}
    - Sub-panels: {len(rep.get("panels",[]))}
    - Chart guess: {rep.get("chart_guess","–")}
    - Palette: {pal}
    - OCR (snippet): {ocr_snip}
    """).strip()

def _build_figure_context(rep: dict) -> str:
    panels_txt=[]
    for idx,p in enumerate(rep.get("panels", [])):
        tag=chr(65+(idx%26)); bb=p.get("bbox"); ex=(p.get("ocr_excerpt") or "").replace("\n"," ").strip()
        ex=textwrap.shorten(ex, width=160, placeholder="…") if ex else ""
        panels_txt.append(f"- Panel {tag}: bbox={bb}, type={p.get('detected_chart','?')}" + (f", OCR: {ex}" if ex else ""))
    pal=_brief_palette_desc(rep.get("dominant_colors", []))
    ocr=(rep.get("ocr_text") or "").strip()
    ocr_short=textwrap.shorten(ocr.replace("\n"," "), width=600, placeholder="…") if ocr else ""
    return textwrap.dedent(f"""
    WIDTH: {rep.get('width')}  HEIGHT: {rep.get('height')}
    SCALE_BAR: {"Yes" if rep.get('has_scale_bar') else "No"}
    CHART_GUESS: {rep.get('chart_guess','')}
    PALETTE: {pal}
    PANELS:
    {chr(10).join(panels_txt) if panels_txt else "- (none)"}
    OCR_FULL_SNIPPET:
    {ocr_short or "(none)"}
    """).strip()

def build_context_by_type(rep: dict, fig_type: str) -> str:
    base=_build_figure_context(rep)
    hints={
        "histology":"Microscopy image (likely H&E). Focus on tissue architecture, nuclei/stroma patterns, and ROIs.",
        "plot":"Scientific plot. Focus on axes, apparent trends, comparisons. Do not invent values.",
        "table":"Table or grid. Describe rows/columns and any headings visible via OCR.",
        "pipeline":"Pipeline/diagram. Describe major blocks, data/control flow, and key stages.",
        "ui":"Software UI/screenshot. Describe visible modules/menus and user flow at a high level.",
        "photo":"Natural photo. Describe salient objects and spatial relationships.",
        "unknown":"Type uncertain. Be cautious and rely strictly on provided cues."
    }[fig_type]
    return f"FIGURE_TYPE: {fig_type}\nTYPE_HINT: {hints}\n\n{base}"

def generate_interpretation_md(rep: dict, model_name: str = None) -> str:
    fig_type, conf, reasons = classify_image_any(rep)
    context = build_context_by_type(rep, fig_type)
    fallback = _rule_based_interpretation(rep, fig_type, conf, reasons)

    try:
        llm = ChatOpenAI(model=(model_name or LLM_MODEL), temperature=0.15, max_tokens=800)
    except Exception:
        return fallback

    system = (
        "You are an expert figure interpreter for science/tech images. "
        "STRICT: Use ONLY provided context; if unknown, say so; never invent numbers/labels. "
        "Map claims to cues in 'Evidence mapping'."
    )
    user = f"""
CLASSIFIER_CONFIDENCE: {conf:.2f}
CLASSIFIER_REASONS: {', '.join(reasons) if reasons else '(none)'}
CONTEXT:
{context}

Return **three sections** in Markdown:
1) **Interpretation** — 4–8 sentences tailored to FIGURE_TYPE. Grounded only in context.
2) **Evidence mapping** — bullet points referencing concrete cues (OCR snippet, axes/lines, palette, ROIs, scale bar, EXIF).
3) **Caption/Legend (draft)** — 3–6 sentences appropriate to FIGURE_TYPE.
"""
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({})
    except Exception:
        return fallback

# ===== Sidebar =====
with st.sidebar:
    st.subheader("Upload files")
    uploads = st.file_uploader("Drop PDF/TXT/DOC/DOCX/XLSX files",
        type=["pdf","txt","md","csv","doc","docx","xlsx","xlsm","xls"],
        accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1: build_btn = st.button("Ingest Files", type="primary")
    with c2: load_btn  = st.button("Load Existing Index")

# ===== Index actions =====
if build_btn:
    if not uploads: st.warning("Please upload at least one file.")
    else:
        with st.spinner("Parsing files, extracting figures & embedding…"):
            docs, figs = load_files_to_documents(uploads)
            if not docs: st.error("No parsable content found."); 
            else:
                try:
                    vs, _, splits, backend = build_index(docs)
                    st.session_state.vectorstore = vs
                    st.session_state.all_docs = splits
                    st.session_state.all_figs = figs
                    st.session_state.backend = backend
                    st.success(f"Indexed {len(splits)} chunks ({backend}). Figures: {len(figs)}.")
                except Exception as e:
                    st.exception(e)

if load_btn:
    with st.spinner("Loading existing index…"):
        vs, _, backend = load_existing_index()
        if vs is not None:
            st.session_state.vectorstore = vs
            st.session_state.backend = backend
            st.info(f"Loaded existing collection with backend: {backend}.")
        else:
            st.warning("No persisted index found; please ingest files.")

# ===== Tabs =====
summary_tab, qa_tab, figs_tab = st.tabs(["📝 Summary", "❓ Q&A", "🖼️ Figures"])

with summary_tab:
    st.write("Generate a concise overview of the ingested corpus.")
    if st.session_state.vectorstore is None and not st.session_state.all_docs:
        st.info("Build or load an index to enable summarization.")
    else:
        if st.button("Generate Summary", key="summ_btn"):
            with st.spinner("Summarizing…"):
                subset = list(st.session_state.all_docs)[:25]
                if not subset: st.warning("No chunks available to summarize."); 
                else:
                    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
                    chain = summary_prompt | llm | StrOutputParser()
                    ctx = "\n\n".join([
                        f"[src={d.metadata.get('source','-')} sheet={d.metadata.get('sheet','-')} page={d.metadata.get('page','-')}]\n{d.page_content}"
                        for d in subset
                    ])
                    st.markdown(chain.invoke({"context": ctx}))

with qa_tab:
    st.write("Ask questions strictly from your files.")
    if st.session_state.vectorstore is None:
        st.info("Build or load an index to enable Q&A.")
    else:
        q = st.text_input("Your question")
        if st.button("Ask", type="primary") and q.strip():
            with st.spinner("Retrieving evidence & answering…"):
                retr = st.session_state.vectorstore.as_retriever(search_type="mmr",
                        search_kwargs={"k": TOP_K, "lambda_mult": MMR_LAMBDA})
                docs = retr.get_relevant_documents(q)
                if not docs: st.warning("No sufficiently relevant context found. I don't know from the provided files.")
                else:
                    ctx = "\n\n".join([
                        f"[src={d.metadata.get('source','-')} sheet={d.metadata.get('sheet','-')} page={d.metadata.get('page','-')}]\n{d.page_content}"
                        for d in docs
                    ])
                    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
                    chain = qa_prompt | llm | StrOutputParser()
                    st.markdown("**Answer**")
                    st.write(chain.invoke({"question": q, "context": ctx}))

                    st.markdown("**Cited Chunks**")
                    for i, d in enumerate(docs, 1):
                        with st.expander(f"{i}. {d.metadata.get('source')} — sheet: {d.metadata.get('sheet') or '-'} — page: {d.metadata.get('page') or '-'}"):
                            st.code(d.page_content[:2000])

with figs_tab:
    st.write("Extracted figures from PDFs (or upload standalone images).")
    lang = st.text_input("OCR language", value="eng", key="fig_lang")
    local_imgs = st.file_uploader("Optional: Upload additional images", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True, key="fig_add_upl")

    figures = list(st.session_state.all_figs)
    if local_imgs:
        for f in local_imgs:
            try: figures.append(Image.open(f).convert("RGB"))
            except Exception: st.warning(f"Could not open {f.name}")

    if not figures: st.info("No figures yet — upload PDFs then click *Ingest Files*, or add images above.")
    else:
        for i, img in enumerate(figures, start=1):
            st.markdown(f"### Figure {i}")
            col1, col2 = st.columns([2,3])
            with col1: st.image(img, caption="Original", use_container_width=True)

            rep = analyze_image(img, lang=lang)
            boxes = [p["bbox"] for p in rep["panels"]]
            overlay = to_cv(img)
            if CV2_OK and overlay is not None and boxes:
                overlay = draw_boxes(overlay, boxes)

            with col2:
                if CV2_OK and overlay is not None:
                    import numpy as _np
                    import cv2 as _cv2
                    st.image(Image.fromarray(_cv2.cvtColor(overlay, _cv2.COLOR_BGR2RGB)), caption="Detected subpanels (A, B, C…)", use_container_width=True)
                else:
                    st.info("Overlay unavailable.")

            k1,k2,k3 = st.columns(3)
            with k1:
                st.subheader("Chart guess"); st.write(rep["chart_guess"])
                st.subheader("Scale bar");  st.write("Yes" if rep["has_scale_bar"] else "No")
            with k2:
                st.subheader("Top colors"); st.image(palette_image(rep["dominant_colors"]), use_container_width=True)
            with k3:
                st.subheader("EXIF / Metadata"); st.json(rep["exif"] or {"note":"no EXIF"})

            with st.expander("OCR (full image)"):
                st.code(rep["ocr_text"] or "(no OCR text)")

            if st.button("Generate interpretation", key=f"gen_interpret_{i}"):
                with st.spinner("Interpreting…"):
                    md = generate_interpretation_md(rep, model_name=None)
                    st.markdown(md)
                    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    st.download_button("⬇️ Download interpretation (Markdown)", data=md.encode("utf-8"),
                        file_name=f"figure_{i}_interpretation_{ts}.md", mime="text/markdown", use_container_width=True)

            st.markdown("---")
