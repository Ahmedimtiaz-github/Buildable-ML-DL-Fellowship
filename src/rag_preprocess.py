import os, sys, json, uuid
from glob import glob
import pandas as pd
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_DIR = os.path.join(ROOT, 'data', 'rag_knowledge')
OUT_DIR = os.path.join(ROOT, 'data', 'processed', 'rag')
os.makedirs(OUT_DIR, exist_ok=True)

texts = []

# read txt files
for path in glob(os.path.join(INPUT_DIR, '**', '*.txt'), recursive=True):
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        content = f.read().strip()
        if not content:
            continue
        # split into lines and keep non-empty
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        texts.extend(lines)

# read CSVs
for path in glob(os.path.join(INPUT_DIR, '**', '*.csv'), recursive=True):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print('Skipping unreadable csv', path, e); continue
    # if 3 or more columns, convert to triplet sentences using first three columns
    if df.shape[1] >= 3:
        cols = df.columns.tolist()
        s = df.apply(lambda r: f"{r[cols[0]]} {r[cols[1]]} {r[cols[2]]}.", axis=1).tolist()
        texts.extend([str(x).strip() for x in s if str(x).strip()])
    else:
        # otherwise join rows
        for row in df.fillna('').astype(str).values:
            cand = ' '.join([c for c in row if c])
            if cand.strip():
                texts.append(cand.strip())

print(f'Collected {len(texts)} text items from RAG knowledge folder.')

# join and chunk into ~200-word chunks (by words)
all_text = '\n'.join(texts)
# simple sentence split (split on . ? ! followed by space)
sentences = re.split(r'(?<=[\.\!\?])\s+', all_text)
chunks = []
current = []
current_word_count = 0
target_words = 200

for s in sentences:
    wc = len(s.split())
    if current_word_count + wc <= target_words:
        current.append(s)
        current_word_count += wc
    else:
        if current:
            chunks.append(' '.join(current).strip())
        current = [s]
        current_word_count = wc
# last
if current:
    chunks.append(' '.join(current).strip())

# save chunks to jsonl
out_path = os.path.join(OUT_DIR, 'chunks.jsonl')
with open(out_path, 'w', encoding='utf8') as f:
    for i,txt in enumerate(chunks):
        obj = {'id': str(uuid.uuid4()), 'text': txt}
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f'Wrote {len(chunks)} chunks to', out_path)
