# ML Challenge – Handoff

## Environment
- Python 3.11
- (Optional GPU) PyTorch / CUDA or ROCm (not required for TF-IDF+Ridge)
- See requirements.lock.txt for the exact versions used

## How to run
1) Create & select a virtual env
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
   - pip install -r requirements.txt
     # or use the pinned file:
     # pip install -r requirements.lock.txt
2) Open notebooks/example.ipynb (or your main notebook)
3) Make sure DATA is in .\data\
4) Run the training / predict cells

## Data
- .\data\train.csv  (75,000 rows)
- .\data\test.csv   (75,000 rows)

## Artifacts (project root)
- .\artifacts\baseline_tfidf_ridge.pkl
- .\artifacts\submission_final.csv

## Model config (best from CV sweep)
- TfidfVectorizer: analyzer='word', ngram_range=(1,2), max_features=300_000
- Ridge: alpha=1.2, random_state=42
- Text column: 'catalog_content_clean' (lowercased, URLs dropped, non-alnum stripped)

## Notes
- Notebook saves artifacts to project-root 'artifacts/' (not notebooks/artifacts)
- Recreate submission: load the .pkl and predict on test -> write artifacts/submission_final.csv
