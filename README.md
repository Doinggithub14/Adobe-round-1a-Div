# Adobe Hackathon – PDF Outline Extractor (Offline)

This project extracts structured outlines like Title, H1, H2, H3 from PDF files using an ML model trained offline — as part of Adobe Hackathon Round 1A.

---

## What We’ve Done So Far

- Extracted font and layout-based features using `PyMuPDF`
- Converted JSON ground truth + PDFs into a dataset (CSV)
- Trained an XGBoost classifier using TF-IDF vectorized features
- Evaluated using precision/recall metrics
- Saved model, vectorizer, and encoder
- Configured everything to run in Docker (offline)

---

## How to Run (Docker)

### Step 1: Build Docker image

```bash
docker build --platform linux/amd64 -t adobe-headings:v1 .