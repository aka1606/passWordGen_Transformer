"""
Telecharge RockYou et prepare le split train/eval.
A lancer sur le serveur avant train_server.py.
Usage: python scripts/prepare_data.py
"""

import os, random, urllib.request, gzip, shutil, time

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
GZ_PATH    = os.path.join(DATA_DIR, 'raw', 'rockyou.txt.gz')
TXT_PATH   = os.path.join(DATA_DIR, 'raw', 'rockyou.txt')
TRAIN_PATH = os.path.join(DATA_DIR, 'splits', 'rockyou_train.txt')
EVAL_PATH  = os.path.join(DATA_DIR, 'splits', 'rockyou_eval.txt')

ROCKYOU_URL = "https://sourceforge.net/projects/wordlist-collection/files/rockyou.txt.gz/download"

def download(url, dest):
    print(f"Telechargement depuis SourceForge...")
    t0 = time.time()
    def progress(count, block, total):
        pct = count * block / total * 100
        if int(pct) % 10 == 0:
            print(f"   {pct:.0f}%", end='\r')
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\n   Telecharge en {time.time()-t0:.0f}s ({os.path.getsize(dest)//1024//1024}MB)")

def decompress(gz, txt):
    print(f"Decompression...")
    with gzip.open(gz, 'rb') as f_in, open(txt, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"   {os.path.getsize(txt)//1024//1024}MB")

def prepare_split(txt, train_out, eval_out, seed=42):
    random.seed(seed)
    print(f"Nettoyage et split...")
    t0 = time.time()

    passwords = set()
    with open(txt, 'rb') as f:
        for line in f:
            try:
                pwd = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                pwd = line.decode('latin-1').strip()
            if pwd and 4 <= len(pwd) <= 30 and all(ord(c) >= 32 for c in pwd):
                passwords.add(pwd)

    passwords = list(passwords)
    random.shuffle(passwords)
    n_eval  = max(10_000, len(passwords) // 100)
    eval_s  = passwords[:n_eval]
    train_s = passwords[n_eval:]

    with open(train_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_s))
    with open(eval_out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_s))

    print(f"   Total: {len(passwords):,} | Train: {len(train_s):,} | Eval: {len(eval_s):,}")
    print(f"   Fait en {time.time()-t0:.0f}s")

os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'splits'), exist_ok=True)

if os.path.exists(TRAIN_PATH):
    print(f"Train deja present ({TRAIN_PATH}), rien a faire.")
else:
    if not os.path.exists(TXT_PATH):
        if not os.path.exists(GZ_PATH):
            download(ROCKYOU_URL, GZ_PATH)
        decompress(GZ_PATH, TXT_PATH)

    prepare_split(TXT_PATH, TRAIN_PATH, EVAL_PATH)
    print(f"\nDonnees pretes:")
    print(f"  Train -> {TRAIN_PATH}")
    print(f"  Eval  -> {EVAL_PATH}")
