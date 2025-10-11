#!/usr/bin/env python
# coding: utf-8

# ##  Basic Library imports

# In[1]:


import os
import pandas as pd 
import numpy as np


# ##  Read Dataset

# In[5]:


from pathlib import Path
import sys, pandas as pd

PROJECT_ROOT = Path(r"D:\amazon ML challenge")
DATASET_DIR  = PROJECT_ROOT / "data"
SRC_DIR      = PROJECT_ROOT / "src"
IMAGES_DIR   = PROJECT_ROOT / "images"   # <-- images will be saved here
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# make 'src' importable (so you can do 'from src.utils import ...' later if needed)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# load data (must include 'image_link' column)
sample_test = pd.read_csv(DATASET_DIR / "sample_test.csv")
print("Rows in sample_test:", len(sample_test))


# In[23]:


from pathlib import Path

def find_root(markers=("requirements.txt", ".git", "data")):
    p = Path.cwd()
    for _ in range(6):
        if any((p / m).exists() for m in markers):
            return p
        p = p.parent
    return Path.cwd()

ROOT = find_root()
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


# In[6]:


# Safe threaded downloader (you can paste this here or put it into src/utils.py)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import pandas as pd, requests, os, re, time
from tqdm import tqdm

def _safe_name(url: str, idx: int) -> str:
    path = urlparse(url).path
    base = os.path.basename(path) or f"img_{idx}.jpg"
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)

def _download_one(url: str, out_dir: Path, idx: int, timeout: int = 15, retries: int = 2):
    if not isinstance(url, str) or not url.strip():
        return False, "empty"
    fname = _safe_name(url, idx)
    dst = out_dir / fname
    if dst.exists():
        return True, "exists"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            return True, "ok"
        except Exception as e:
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
            else:
                return False, str(e)

def download_images(urls, out_dir: str | Path, max_workers: int = 12, timeout: int = 15, retries: int = 2):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ser = pd.Series(urls).dropna().astype(str).str.strip()
    ser = ser[ser.ne("")].reset_index(drop=True)
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download_one, url, out_dir, i, timeout, retries): i for i, url in ser.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            success, msg = fut.result()
            if success and msg == "exists": skip += 1
            elif success: ok += 1
            else: fail += 1
    print(f"Done. ok={ok}, skipped(existing)={skip}, failed={fail}, saved to {out_dir.resolve()}")


# In[7]:


download_images(sample_test["image_link"], IMAGES_DIR, max_workers=12)


# In[11]:


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\amazon ML challenge")
DATASET_DIR  = PROJECT_ROOT / "data"

assert (DATASET_DIR / "train.csv").exists(), f"Missing: {DATASET_DIR/'train.csv'}"
assert (DATASET_DIR / "test.csv").exists(),  f"Missing: {DATASET_DIR/'test.csv'}"

train = pd.read_csv(DATASET_DIR / "train.csv")
test  = pd.read_csv(DATASET_DIR / "test.csv")

print("Loaded shapes -> train:", train.shape, "| test:", test.shape)
print("Columns:", list(train.columns)[:12], "...")


# In[12]:


train.head()


# In[13]:


train.info()


# In[14]:


import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

PROJECT_ROOT = Path(r"D:\amazon ML challenge")
DATASET_DIR  = PROJECT_ROOT / "data"
ART_DIR      = PROJECT_ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)

text_col = "catalog_content"
target   = "price"

# (You already have these loaded, but keeping them here makes the cell self-contained)
train = pd.read_csv(DATASET_DIR / "train.csv")
test  = pd.read_csv(DATASET_DIR / "test.csv")
train[text_col] = train[text_col].fillna("")
test[text_col]  = test[text_col].fillna("")
print(train.shape, test.shape)


# In[18]:


# Mini hyperparam sweep: TF-IDF + Ridge
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
import re

# Reuse your chosen text source; if you used A/B, set text_col accordingly
TEXT_COL = "catalog_content_clean" if "catalog_content_clean" in train.columns else "catalog_content"

# Safer cleaner (handles NaNs)
def _clean_series(series: pd.Series) -> pd.Series:
    series = series.fillna("").astype(str).str.lower()
    series = series.str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
    series = series.str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    return series.str.replace(r"\s+", " ", regex=True).str.strip()

# Only used if your data isn't pre-cleaned:
CLEAN_IN_PIPE = (TEXT_COL == "catalog_content")

def make_pipe(max_features=200_000, ngram_hi=2, analyzer="word", alpha=1.0):
    steps = []
    if CLEAN_IN_PIPE:
        steps.append(("clean", FunctionTransformer(_clean_series, validate=False)))
    steps += [
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_hi),
            analyzer=analyzer
        )),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ]
    return Pipeline(steps)

param_grid = [
    # Word-level (fast, strong baseline)
    {"max_features": 150_000, "ngram_hi": 2, "analyzer": "word", "alpha": 0.8},
    {"max_features": 200_000, "ngram_hi": 2, "analyzer": "word", "alpha": 1.0},
    {"max_features": 300_000, "ngram_hi": 2, "analyzer": "word", "alpha": 1.2},
    # Light char-grams often help messy text
    {"max_features": 300_000, "ngram_hi": 5, "analyzer": "char_wb", "alpha": 1.0},
]

X = train[TEXT_COL]
y = train["price"].astype(float)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for cfg in param_grid:
    maes = []
    for tr_idx, va_idx in cv.split(X):
        pipe = make_pipe(**cfg)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[va_idx])
        maes.append(mean_absolute_error(y.iloc[va_idx], pred))
    results.append({**cfg, "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))})

pd.DataFrame(results).sort_values("cv_mae_mean")


# In[20]:


# Final training with best hyperparameters
from joblib import dump

best_cfg = {"max_features": 300_000, "ngram_hi": 2, "analyzer": "word", "alpha": 1.2}
final_pipe = make_pipe(**best_cfg)

TEXT_COL = "catalog_content_clean" if "catalog_content_clean" in train.columns else "catalog_content"
final_pipe.fit(train[TEXT_COL], train["price"])

from pathlib import Path

# create artifacts dir relative to your notebook's working dir
Path("artifacts").mkdir(parents=True, exist_ok=True)

# now your original lines work
from joblib import dump
dump(final_pipe, "artifacts/baseline_tfidf_ridge.pkl")

test_pred = final_pipe.predict(test[TEXT_COL])
sub = pd.DataFrame({"sample_id": test["sample_id"], "price": test_pred})
sub.to_csv("artifacts/submission_final.csv", index=False)

print("Saved: artifacts/submission_final.csv")


# In[22]:


from pathlib import Path
import pandas as pd
from joblib import load

# SHOW where the notebook is running
print("Notebook CWD:", Path.cwd())

# Try to locate any 'submission_final.csv' under the project
hits = list(Path.cwd().rglob("submission_final.csv"))
print("Found:", [str(p) for p in hits])

# Robust project-root detection (looks upward for markers)
def find_root(markers=("requirements.txt", ".git", "data")):
    p = Path.cwd()
    for _ in range(6):
        if any((p / m).exists() for m in markers):
            return p
        p = p.parent
    return Path.cwd()

ROOT = find_root()
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

print("Resolved ROOT:", ROOT)
print("Saving artifacts to:", ART.resolve())

# If you don't have test loaded, reload
try:
    _ = test.head()
except NameError:
    import pandas as pd
    test = pd.read_csv(ROOT / "data" / "test.csv")

# Decide text column
TEXT_COL = "catalog_content_clean" if "catalog_content_clean" in test.columns else "catalog_content"

# Load model and re-save submission to the resolved artifacts path
pipe = load(ART / "baseline_tfidf_ridge.pkl")  # model you already have
pred = pipe.predict(test[TEXT_COL])
sub = pd.DataFrame({"sample_id": test["sample_id"], "price": pred})
out_path = ART / "submission_final.csv"
sub.to_csv(out_path, index=False)

print("WROTE:", out_path.resolve(), "| rows:", len(sub))


# In[15]:


X_tr, X_va, y_tr, y_va = train_test_split(
    train[text_col], train[target],
    test_size=0.2, random_state=42
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1,2),
        lowercase=True,
        strip_accents="unicode",
        min_df=2
    )),
    ("ridge", Ridge(alpha=1.0, random_state=42))
])

pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_va)
mae = mean_absolute_error(y_va, pred)
print(f"Holdout MAE: {mae:.5f}")

joblib.dump(pipe, ART_DIR / "baseline_tfidf_ridge.pkl")
print("Saved:", ART_DIR / "baseline_tfidf_ridge.pkl")


# In[16]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1,2),
        lowercase=True,
        strip_accents="unicode",
        min_df=2
    )),
    ("ridge", Ridge(alpha=1.0, random_state=42))
])

cv_scores = -cross_val_score(
    cv_pipe,
    train[text_col], train[target],
    scoring="neg_mean_absolute_error",
    cv=kf,
    n_jobs=-1
)
print("CV MAE per fold:", [round(s,5) for s in cv_scores])
print("CV MAE mean:", cv_scores.mean(), "±", cv_scores.std())


# In[17]:


final_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=200_000,
        ngram_range=(1,2),
        lowercase=True,
        strip_accents="unicode",
        min_df=2
    )),
    ("ridge", Ridge(alpha=1.0, random_state=42))
])

final_pipe.fit(train[text_col], train[target])
joblib.dump(final_pipe, ART_DIR / "final_tfidf_ridge.pkl")

test_pred = final_pipe.predict(test[text_col])

# Build submission (match the sample output schema if needed)
sub = pd.DataFrame({
    "sample_id": test["sample_id"],
    "price": test_pred
})
sub_path = ART_DIR / "submission_baseline.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
sub.head()


# In[10]:


from PIL import Image
import matplotlib.pyplot as plt
import random, os

samples = random.sample(os.listdir(IMAGES_DIR), 5)
for s in samples:
    img = Image.open(IMAGES_DIR / s)
    plt.imshow(img)
    plt.title(s)
    plt.show()


# In[3]:


# Try to import from your project utils; define a minimal fallback if not present
try:
    from src.utils import download_images
    import src.utils as _u
    print("utils loaded from:", _u.__file__)
except Exception as e:
    print("Could not import src.utils.download_images:", e, "\nDefining a minimal version here.")
    import re, time, requests, os
    from urllib.parse import urlparse
    def download_images(urls, out_dir, timeout=15):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        ok = bad = 0
        for i, u in enumerate(urls):
            if not isinstance(u, str) or not u.strip():
                bad += 1; continue
            path = urlparse(u).path
            base = os.path.basename(path) or f"img_{i}.jpg"
            base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
            dst = out_dir / base
            if dst.exists(): ok += 1; continue
            try:
                r = requests.get(u, timeout=timeout, stream=True)
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk: f.write(chunk)
                ok += 1
            except Exception:
                bad += 1; time.sleep(0.2)
        print(f"Downloaded: {ok}, Failed: {bad}, Saved to: {out_dir.resolve()}")


# In[1]:


from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import pandas as pd, requests, os, re, time
from tqdm import tqdm

def _safe_name(url: str, idx: int) -> str:
    path = urlparse(url).path
    base = os.path.basename(path) or f"img_{idx}.jpg"
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)

def _download_one(url: str, out_dir: Path, idx: int, timeout: int = 15, retries: int = 2):
    if not isinstance(url, str) or not url.strip():
        return False, "empty"
    fname = _safe_name(url, idx)
    dst = out_dir / fname
    if dst.exists():
        return True, "exists"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return True, "ok"
        except Exception as e:
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
            else:
                return False, str(e)

def download_images(urls, out_dir: str | Path, max_workers: int = 12, timeout: int = 15, retries: int = 2):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Clean the input a bit
    ser = pd.Series(urls).dropna().astype(str).str.strip()
    ser = ser[ser.ne("")].reset_index(drop=True)

    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download_one, url, out_dir, i, timeout, retries): i for i, url in ser.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            success, msg = fut.result()
            if success and msg == "exists":
                skip += 1
            elif success:
                ok += 1
            else:
                fail += 1
    print(f"Done. ok={ok}, skipped(existing)={skip}, failed={fail}, saved to {out_dir.resolve()}")


# In[4]:


# Make sure the column exists
assert "image_link" in sample_test.columns, f"'image_link' column not found. Got: {list(sample_test.columns)[:10]}"
download_images(sample_test["image_link"], str(IMAGES_DIR))


# In[12]:


#DATASET_FOLDER = '../dataset/'
#train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
#test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))


# In[7]:


from utils import download_images
download_images(sample_test['image_link'], '../images')


# In[14]:


assert len(os.listdir('../images')) > 0


# In[ ]:


rm -rf ../images

