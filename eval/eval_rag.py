# eval/eval_rag.py
import os, json, argparse, pandas as pd
import numpy as np
from collections import defaultdict

def mrr_score(ranks):
    # ranks: list of 1-based ranks or None
    vals = []
    for r in ranks:
        if r is None:
            vals.append(0.0)
        else:
            vals.append(1.0/float(r))
    return float(np.mean(vals))

def main(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    # Expect test file with columns: query, relevant_chunk_id (or relevant_text)
    test_csv = args.test or "data/rag_test.csv"
    if not os.path.exists(test_csv):
        print("Test file not found:", test_csv)
        return
    df = pd.read_csv(test_csv)
    if 'query' not in df.columns or ('relevant_chunk_id' not in df.columns and 'relevant_text' not in df.columns):
        print("Test CSV requires 'query' and one of 'relevant_chunk_id' or 'relevant_text'.")
        return
    # Load retrieval function: we'll call the module src.rag_tool.retrieve if available
    try:
        from src.rag_tool import retrieve
    except Exception as e:
        print("Could not import src.rag_tool.retrieve:", e)
        return

    ks = args.topk or 5
    ranks = []
    hits = 0
    records = []
    for _, row in df.iterrows():
        q = row['query']
        relevant_id = row.get('relevant_chunk_id', None)
        relevant_text = row.get('relevant_text', None)
        retrieved = retrieve(q, top_k=ks)
        ids = [r['id'] for r in retrieved]
        # compute rank
        rank = None
        if relevant_id and relevant_id in ids:
            rank = ids.index(relevant_id) + 1
        else:
            # fallback: check if relevant_text appears in snippets
            if relevant_text:
                found = False
                for i, r in enumerate(retrieved):
                    if relevant_text.strip().lower() in r['text_snippet'].strip().lower():
                        rank = i+1
                        found = True
                        break
        ranks.append(rank)
        if rank is not None:
            hits += 1
        records.append({"query": q, "retrieved_ids": ids, "rank": rank})
    hit_rate = hits / len(df)
    mrr = mrr_score(ranks)
    out_metrics = {"hit_rate": float(hit_rate), "mrr": float(mrr), "n": int(len(df))}
    with open(os.path.join(out, "rag_metrics.json"), "w") as fh:
        json.dump(out_metrics, fh, indent=2)
    pd.DataFrame(records).to_csv(os.path.join(out,"rag_examples.csv"), index=False)
    print("Saved rag metrics and examples.")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--out", default="reports/eval/rag")
    args = parser.parse_args()
    main(args)
