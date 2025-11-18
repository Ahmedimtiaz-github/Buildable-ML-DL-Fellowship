# eval/eval_agent.py
import os, json, argparse, subprocess, sys, time, csv
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def call_agent_cli(query, venv_python):
    # calls main.py --cli "<query>"
    proc = subprocess.run([venv_python, "main.py", "--cli", query], capture_output=True, text=True)
    # Expect CLI prints a JSON dict line; try to parse last line
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    # find a JSON-like line
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except:
                continue
    return None

def main(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    test_csv = args.test or "data/agent_test.csv"
    if not os.path.exists(test_csv):
        print("Test file not found:", test_csv)
        return
    df = pd.read_csv(test_csv)
    if 'query' not in df.columns or 'gold_tool' not in df.columns:
        print("Test CSV requires columns 'query' and 'gold_tool'")
        return
    venv_python = args.venv or ".\\venv\\Scripts\\python.exe"
    preds = []
    for _, row in df.iterrows():
        q = row['query']
        res = call_agent_cli(q, venv_python)
        if res is None:
            preds.append(None)
        else:
            preds.append(res.get('tool_used'))
        time.sleep(0.2)
    df['pred_tool'] = preds
    df.to_csv(os.path.join(out,"agent_results.csv"), index=False)
    # compute metrics
    df2 = df.dropna(subset=['pred_tool'])
    acc = accuracy_score(df2['gold_tool'], df2['pred_tool'])
    cm = confusion_matrix(df2['gold_tool'], df2['pred_tool'], labels=sorted(list(set(df2['gold_tool']) | set(df2['pred_tool']))))
    with open(os.path.join(out,'agent_metrics.json'),'w') as fh:
        json.dump({"accuracy": float(acc)}, fh, indent=2)
    pd.DataFrame(cm).to_csv(os.path.join(out,'agent_confusion_matrix.csv'), index=False)
    print("Saved agent results and metrics.")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=None)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--out", default="reports/eval/agent")
    args = parser.parse_args()
    main(args)
