"""
Prépare le dataset d'entraînement combiné.

Sources:
  - RockYou brut (data/raw/rockyou.txt)
  - RockYou avec fréquences (data/extra/rockyou-withcount.txt)
  - Pwdb top 10M (data/extra/pwdb_top10M.txt)
  - 000webhost (data/extra/000webhost.txt)
  - phpbb (data/extra/phpbb.txt)

Sortie:
  - data/splits/combined_train.txt  → entraînement (dédupliqué ou freq-weighted)
  - data/splits/combined_eval.txt   → évaluation (143k passwords, inchangé)

Usage:
    python scripts/preprocess/prepare_dataset.py
    python scripts/preprocess/prepare_dataset.py --freq-weight --max-repeat 10
"""

import os, argparse, random, re
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCES = {
    'rockyou':        os.path.join(BASE_DIR, 'data', 'raw', 'rockyou.txt'),
    'rockyou_wcount': os.path.join(BASE_DIR, 'data', 'extra', 'rockyou-withcount.txt'),
    'pwdb_10m':       os.path.join(BASE_DIR, 'data', 'extra', 'pwdb_top10M.txt'),
    '000webhost':     os.path.join(BASE_DIR, 'data', 'extra', '000webhost.txt'),
    'phpbb':          os.path.join(BASE_DIR, 'data', 'extra', 'phpbb.txt'),
}

EVAL_FILE   = os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_eval.txt')
OUT_TRAIN   = os.path.join(BASE_DIR, 'data', 'splits', 'combined_train.txt')
OUT_EVAL    = os.path.join(BASE_DIR, 'data', 'splits', 'combined_eval.txt')

MIN_LEN, MAX_LEN = 4, 30


def is_valid(p):
    return p and MIN_LEN <= len(p) <= MAX_LEN and p.isprintable()


def load_rockyou_withcount(path, max_repeat=10):
    """Charge rockyou-withcount.txt: format 'COUNT PASSWORD'"""
    passwords = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                count_str, pwd = parts
                try:
                    count = int(count_str)
                except ValueError:
                    pwd = line
                    count = 1
            else:
                pwd = line
                count = 1
            if not is_valid(pwd):
                continue
            # Répéter proportionnellement à la fréquence, cappé à max_repeat
            repeats = min(max_repeat, max(1, round(1 + (count / 10000))))
            for _ in range(repeats):
                passwords.append(pwd)
    return passwords


def load_plain(path):
    """Charge un fichier de passwords un par ligne."""
    passwords = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if is_valid(pwd) and not pwd.startswith('#'):
                passwords.append(pwd)
    return passwords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq-weight', action='store_true',
                        help='Pondérer par fréquence (rockyou-withcount)')
    parser.add_argument('--max-repeat', type=int, default=5,
                        help='Répétitions max pour freq-weighting (défaut: 5)')
    parser.add_argument('--no-extra', action='store_true',
                        help='Utiliser seulement RockYou')
    args = parser.parse_args()

    print("=" * 60)
    print("PRÉPARATION DU DATASET COMBINÉ")
    print("=" * 60)

    # Charger l'eval existant (inchangé)
    eval_pwds = set()
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            eval_pwds = {l.strip() for l in f if l.strip()}
        print(f"\nEval set: {len(eval_pwds):,} passwords (conservé tel quel)")

    # Charger les sources
    all_train = []

    # 1. RockYou (source principale)
    if args.freq_weight and os.path.exists(SOURCES['rockyou_wcount']):
        print(f"\nChargement RockYou avec fréquences...")
        ry = load_rockyou_withcount(SOURCES['rockyou_wcount'], args.max_repeat)
        print(f"  → {len(ry):,} entrées (freq-weighted)")
        all_train.extend(ry)
    elif os.path.exists(SOURCES['rockyou']):
        print(f"\nChargement RockYou brut...")
        ry = load_plain(SOURCES['rockyou'])
        print(f"  → {len(ry):,} passwords")
        all_train.extend(ry)

    if not args.no_extra:
        # 2. Pwdb top 10M
        if os.path.exists(SOURCES['pwdb_10m']):
            print(f"Chargement Pwdb top 10M...")
            pwdb = load_plain(SOURCES['pwdb_10m'])
            print(f"  → {len(pwdb):,} passwords")
            all_train.extend(pwdb)

        # 3. 000webhost
        if os.path.exists(SOURCES['000webhost']):
            print(f"Chargement 000webhost...")
            wh = load_plain(SOURCES['000webhost'])
            print(f"  → {len(wh):,} passwords")
            all_train.extend(wh)

        # 4. phpbb
        if os.path.exists(SOURCES['phpbb']):
            print(f"Chargement phpbb...")
            pb = load_plain(SOURCES['phpbb'])
            print(f"  → {len(pb):,} passwords")
            all_train.extend(pb)

    print(f"\nTotal brut: {len(all_train):,}")

    # Retirer les passwords qui sont dans l'eval
    print(f"Filtrage (retrait eval + doublons)...")
    if not args.freq_weight:
        # Sans freq-weighting: dédupliquer
        all_train = list({p for p in all_train if p not in eval_pwds})
    else:
        # Avec freq-weighting: garder les doublons mais retirer ceux dans eval
        all_train = [p for p in all_train if p not in eval_pwds]

    print(f"Total après filtrage: {len(all_train):,}")

    # Shuffle
    random.shuffle(all_train)

    # Sauvegarder
    os.makedirs(os.path.dirname(OUT_TRAIN), exist_ok=True)

    with open(OUT_TRAIN, 'w', encoding='utf-8') as f:
        for p in all_train:
            f.write(p + '\n')
    print(f"\nSauvegardé: {OUT_TRAIN}")
    print(f"  {len(all_train):,} passwords d'entraînement")

    # Copier eval
    import shutil
    shutil.copy2(EVAL_FILE, OUT_EVAL)
    print(f"Eval copié: {OUT_EVAL}")

    # Stats
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"  Train: {len(all_train):,}")
    print(f"  Eval:  {len(eval_pwds):,}")
    print(f"  Mode:  {'freq-weighted' if args.freq_weight else 'dédupliqué'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
