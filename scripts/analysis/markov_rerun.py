# -*- coding: utf-8 -*-
"""
Relance Markov (ordres 1-4) sur rockyou_train/eval pour comparaison equitable
avec le Transformer v7 (meme eval set de 143k mots de passe).
"""

import json, random, math, os, time
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_train.txt')
EVAL_PATH  = os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_eval.txt')
OUT_PATH   = os.path.join(BASE_DIR, 'output', 'results', 'markov_results_v2.json')

MAX_TRAIN   = 2_000_000   # 2M suffisent pour Markov, evite 14M inutiles
GEN_COUNT   = 500_000
MAX_LENGTH  = 20
START, END  = '\x00', '\x01'

random.seed(42)


def load(path, limit=None):
    passwords = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if pwd and 4 <= len(pwd) <= 30:
                passwords.append(pwd)
                if limit and len(passwords) >= limit:
                    break
    return passwords


class MarkovChain:
    def __init__(self, order):
        self.order = order
        self.trans = defaultdict(Counter)
        self.totals = defaultdict(int)

    def train(self, passwords):
        for pwd in passwords:
            padded = (START * self.order) + pwd + END
            for i in range(len(padded) - self.order):
                s = padded[i:i+self.order]
                c = padded[i+self.order]
                self.trans[s][c] += 1
                self.totals[s] += 1

    def generate_one(self):
        state = START * self.order
        pwd = []
        for _ in range(MAX_LENGTH):
            if state not in self.trans:
                break
            total = self.totals[state]
            r = random.randint(1, total)
            cum = 0
            chosen = None
            for ch, cnt in self.trans[state].items():
                cum += cnt
                if cum >= r:
                    chosen = ch
                    break
            if chosen is None or chosen == END:
                break
            pwd.append(chosen)
            state = state[1:] + chosen
        return ''.join(pwd)

    def generate(self, n):
        result = set()
        attempts = 0
        while len(result) < n and attempts < n * 10:
            pwd = self.generate_one()
            if len(pwd) >= 4:
                result.add(pwd)
            attempts += 1
        return result

    def perplexity(self, passwords, sample=2000):
        sample_pwds = random.sample(passwords, min(sample, len(passwords)))
        total_lp, total_chars, valid = 0.0, 0, 0
        for pwd in sample_pwds:
            padded = (START * self.order) + pwd + END
            lp = 0.0
            ok = True
            for i in range(len(padded) - self.order):
                s = padded[i:i+self.order]
                c = padded[i+self.order]
                if s not in self.trans or c not in self.trans[s]:
                    ok = False; break
                lp += math.log(self.trans[s][c] / self.totals[s])
            if ok:
                total_lp += lp
                total_chars += len(pwd) + 1
                valid += 1
        if valid == 0 or total_chars == 0:
            return float('inf')
        return round(math.exp(-total_lp / total_chars), 4)


def main():
    print("=" * 60)
    print("MARKOV RERUN - eval set 143k (comparable Transformer v7)")
    print("=" * 60)

    print(f"\nChargement train (max {MAX_TRAIN:,})...")
    train = load(TRAIN_PATH, limit=MAX_TRAIN)
    print(f"  Train charge : {len(train):,}")

    print("Chargement eval (143k)...")
    eval_pwds = load(EVAL_PATH)
    eval_set  = set(eval_pwds)
    print(f"  Eval charge  : {len(eval_set):,}")

    results = {}

    for order in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"MARKOV ORDRE {order}")
        print(f"{'='*60}")

        t0 = time.time()
        m = MarkovChain(order)
        m.train(train)
        print(f"  Entraine en {time.time()-t0:.1f}s  |  etats uniques: {len(m.trans):,}")

        perp_train = m.perplexity(train)
        perp_eval  = m.perplexity(eval_pwds)
        print(f"  Perplexite train: {perp_train}  |  eval: {perp_eval}")

        print(f"  Generation {GEN_COUNT:,} mots de passe...")
        t0 = time.time()
        generated = m.generate(GEN_COUNT)
        print(f"  Generes uniques: {len(generated):,}  ({time.time()-t0:.1f}s)")

        matches = generated & eval_set
        coverage = round(len(matches) / len(eval_set) * 100, 4)
        print(f"  Coverage : {len(matches)}/{len(eval_set)} = {coverage}%")

        results[f'order_{order}'] = {
            'unique_states': len(m.trans),
            'perplexity_train': perp_train,
            'perplexity_eval': perp_eval,
            'generated_unique': len(generated),
            'matches': len(matches),
            'coverage_pct': coverage,
        }

    print(f"\n{'='*60}")
    print("TABLEAU COMPARATIF")
    print(f"{'='*60}")
    print(f"{'Ordre':<8} {'Etats':<10} {'Perp.Train':<13} {'Perp.Eval':<12} {'Coverage':<10}")
    print("-" * 55)
    for o in [1, 2, 3, 4]:
        r = results[f'order_{o}']
        print(f"{o:<8} {r['unique_states']:<10,} {r['perplexity_train']:<13} {r['perplexity_eval']:<12} {r['coverage_pct']}%")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump({'eval_size': len(eval_set), 'train_size': len(train), 'results': results}, f, indent=2)
    print(f"\nSauvegarde : {OUT_PATH}")


if __name__ == '__main__':
    main()
