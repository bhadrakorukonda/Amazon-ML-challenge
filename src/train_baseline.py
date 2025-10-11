import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

def main(a):
    df = pd.read_csv(a.train_csv)
    if a.text_col not in df.columns or a.target_col not in df.columns:
        raise SystemExit(f"Columns not found. Have: {df.columns.tolist()}")

    df[a.text_col] = df[a.text_col].fillna("")
    cat_cols = [c for c in a.cat_cols.split(",") if c and c in df.columns]
    num_cols = [c for c in a.num_cols.split(",") if c and c in df.columns]

    text_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=a.max_features, ngram_range=(1,2)))])
    transformers = [("text", text_pipe, a.text_col)]
    if cat_cols: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols: transformers.append(("num", "passthrough", num_cols))

    pre = ColumnTransformer(transformers)
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=a.alpha))])

    tr, va = train_test_split(df, test_size=0.2, random_state=42)
    pipe.fit(tr[[a.text_col] + cat_cols + num_cols], tr[a.target_col])
    pred = pipe.predict(va[[a.text_col] + cat_cols + num_cols])
    mae = mean_absolute_error(va[a.target_col], pred)
    print(f"MAE: {mae:.6f}")

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "baseline_tfidf_ridge.pkl")
    joblib.dump(pipe, path)
    print(f"Saved {path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/train.csv")
    p.add_argument("--text_col", default="catalog_content")
    p.add_argument("--target_col", default="price")
    p.add_argument("--cat_cols", default="brand")     # comma-separated
    p.add_argument("--num_cols", default="quantity")  # comma-separated
    p.add_argument("--max_features", type=int, default=200_000)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out_dir", default="artifacts")
    main(p.parse_args())
