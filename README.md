
# 🧬 Curation Assistant Tool

A Streamlit app for **bioinformatics curation** that combines:
- 📄 Document summarization & Q&A (PDF, DOCX, TXT, XLSX)
- 🖼️ Figure extraction from PDFs (with OCR text search)
- 📊 Gene frequency calculation from supplementary tables
- 🔎 Gene lookup across cBioPortal-style files (`data_mutations.txt`, `data_cna.txt`, `data_sv.txt`)

---

## Quickstart

### 1. Clone this repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create environment & install dependencies
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Set your OpenAI API key
You’ll need an API key from [OpenAI](https://platform.openai.com/).

```bash
export OPENAI_API_KEY="sk-..."
```

*(On Windows PowerShell:)*  
```powershell
$env:OPENAI_API_KEY="sk-..."
```

### 4. Run the app
```bash
streamlit run new_app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

---

##  Deployment on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Connect your repo and select **`new_app.py`** as the app file.
4. Add your **OpenAI API key** as a **Secret** in Streamlit Cloud (`OPENAI_API_KEY`).
5. Deploy — you’ll get a shareable public link like:

```
https://your-username-your-repo.streamlit.app
```

---

## 📂 File Structure
```
your-repo/
├── new_app.py          # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── runtime.txt         # (optional) Python version for Streamlit Cloud
├── apt.txt             # (optional) system packages, e.g. for OCR
└── .streamlit/
    └── config.toml     # (optional) custom Streamlit theme
```

---

##  Features

- **Summarization & Q&A**
  - Grounded strictly in uploaded files
  - Supports PDFs, Word, TXT, Excel
  - Cites file name, sheet, and page numbers

- **Figure Extraction**
  - Extracts embedded images from PDFs
  - OCR text indexing for searchable captions

- **Gene Frequencies**
  - Upload supplementary tables (.csv/.tsv/.txt/.xlsx)
  - Compute frequencies automatically or with custom denominator
  - Download results as CSV

- **Gene Lookup**
  - Search for single or multiple genes (e.g., `TP53, EGFR, KRAS`)
  - Supports `Hugo_Symbol` in mutation/CNA files
  - Supports `Site1_Hugo_Symbol` / `Site2_Hugo_Symbol` in SV files
  - Reports per-file sample counts and percentages

---

##  Requirements
- Python **3.10+**
- Internet access (for OpenAI API calls)

Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🙋 Usage Example
1. Upload your cBioPortal supplementary files:
   - `data_mutations.txt`
   - `data_cna.txt`
   - `data_sv.txt`
   - `data_clinical_sample.txt`

2. Query genes:
   ```
   TP53, EGFR, KRAS
   ```

3. See sample counts & % frequencies directly in the app.

---

## 👨‍💻 Author
Built by **Baby Anusha Satravada** · Bioinformatics Software Engineer  
Memorial Sloan Kettering Cancer Center (MSK)

---
