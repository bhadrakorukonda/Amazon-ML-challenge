import argparse, pandas as pd, numpy as np, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def main(a):
    df = pd.read_csv(a.train_csv)
    df[a.text_col] = df[a.text_col].fillna("")
    if a.target_col not in df.columns:
        raise SystemExit(f"Missing target column '{a.target_col}'")

    df = df[[a.text_col, a.target_col]].rename(columns={a.target_col: "labels"})
    df["labels"] = df["labels"].astype(float)
    ds = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)

    tok = AutoTokenizer.from_pretrained(a.model)
    def enc(batch): return tok(batch[a.text_col], truncation=True, padding="max_length", max_length=256)
    ds = ds.map(enc, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, problem_type="regression")
    args = TrainingArguments(
        output_dir=a.out_dir, learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=2, evaluation_strategy="epoch",
        logging_steps=50, save_strategy="epoch",
        fp16=torch.cuda.is_available()
    )
    tr = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
    tr.train()
    print(tr.evaluate())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--text_col", default="catalog_content")
    ap.add_argument("--target_col", default="price")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out_dir", default="artifacts_hf")
    main(ap.parse_args())
