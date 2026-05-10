"""
Rules Engine — mutations sur les passwords générés.

Usage:
    python scripts/analysis/rules_engine.py --input output/generated/v7_generated.txt
    python scripts/analysis/rules_engine.py --input output/generated/v7_generated.txt --eval data/splits/rockyou_eval.txt
"""

import os, argparse, time, random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEET_MAP = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0',
    's': '$', 't': '7', 'b': '6', 'g': '9', 'l': '1',
}

LEET_MAP_LIGHT = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$',
}

SUFFIXES = [
    '1', '2', '3', '12', '21', '123', '1234', '12345', '123456',
    '0', '01', '007', '11', '22', '33', '00', '99', '69', '88',
    '2024', '2023', '2022', '2021', '2020', '2019', '2018',
    '24', '23', '22', '21', '20', '19', '18',
    '1990', '1991', '1992', '1993', '1994', '1995',
    '1996', '1997', '1998', '1999', '2000', '2001',
    '!', '!!', '!!!', '!1', '1!', '!123', '123!',
    '.', '*', '#', '@', '?', '_', '-',
    '1!', '1!!', '123!', '!@#', '1@3',
]

PREFIXES = ['1', '12', '123', '0', '00', 'the', 'my', 'mr', 'ms']

MIN_LEN = 4
MAX_LEN = 30


def is_valid(pwd):
    return MIN_LEN <= len(pwd) <= MAX_LEN and pwd.isprintable()


def leet(pwd, leet_map):
    return ''.join(leet_map.get(c.lower(), c) for c in pwd)


def generate_variants(pwd):
    if not pwd or not is_valid(pwd):
        return
    cap = pwd.capitalize()
    low = pwd.lower()
    upp = pwd.upper()
    toggled = ''.join(c.upper() if j % 2 == 0 else c.lower() for j, c in enumerate(pwd))
    leet_full  = leet(pwd, LEET_MAP)
    leet_light = leet(pwd, LEET_MAP_LIGHT)
    leet_cap   = leet(cap, LEET_MAP_LIGHT)

    variants = [pwd, cap, low, upp, leet_full, leet_light, leet_cap, toggled]
    if len(pwd) >= 2:
        variants.append(pwd[:-1] + pwd[-1].upper())
    for s in SUFFIXES:
        variants += [pwd + s, cap + s, low + s]
    for s in ['1', '123', '!', '1!', '2024', '2023']:
        variants += [leet_light + s, leet_cap + s]
    for p in PREFIXES:
        variants += [p + pwd, p + cap]
    if len(pwd) <= 8:
        variants += [pwd + pwd, cap + pwd]

    for v in variants:
        if is_valid(v):
            yield v


def apply_rules(base_passwords, verbose=True, eval_set=None):
    t0 = time.time()
    total = len(base_passwords)
    total_candidates = 0

    if verbose:
        print(f"   Base: {total:,} passwords")
        print(f"   Application des regles...")

    if eval_set is not None:
        # Mode streaming : on check contre eval à la volée pour garder la RAM constante
        matches = set()
        for i, pwd in enumerate(base_passwords):
            for v in generate_variants(pwd):
                total_candidates += 1
                if v in eval_set:
                    matches.add(v)
            if verbose and (i + 1) % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"   {i+1:,}/{total:,} ({(i+1)/total*100:.0f}%) | "
                      f"{total_candidates:,} candidats | {elapsed:.0f}s")
        if verbose:
            print(f"   Done en {time.time()-t0:.0f}s -> {total_candidates:,} candidats, {len(matches):,} matches")
        return matches, total_candidates
    else:
        candidates = set()
        for i, pwd in enumerate(base_passwords):
            for v in generate_variants(pwd):
                candidates.add(v)
            if verbose and (i + 1) % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"   {i+1:,}/{total:,} ({(i+1)/total*100:.0f}%) | "
                      f"{len(candidates):,} candidats | {elapsed:.0f}s")
        if verbose:
            print(f"   Done en {time.time()-t0:.0f}s -> {len(candidates):,} candidats uniques")
        return candidates


def evaluate_coverage(candidates, eval_passwords, label=""):
    eval_set = set(eval_passwords)
    matches  = candidates & eval_set
    pct      = len(matches) / len(eval_set) * 100 if eval_set else 0
    print(f"\n   [{label}] Coverage: {len(matches):,}/{len(eval_set):,} = {pct:.2f}%")
    if matches:
        examples = random.sample(sorted(matches), min(10, len(matches)))
        print(f"   Exemples: {examples}")
    return pct, matches


def coverage_by_rule(base_passwords, eval_passwords, sample=50_000):
    eval_set = set(eval_passwords)
    sample_base = list(set(base_passwords))[:sample]

    print("\n   Contribution par regle (sample):")
    rules = {
        'original':    lambda p: {p},
        'capitalize':  lambda p: {p.capitalize()},
        'upper':       lambda p: {p.upper()},
        'leet_light':  lambda p: {leet(p, LEET_MAP_LIGHT)},
        'leet_full':   lambda p: {leet(p, LEET_MAP)},
        'suffix_123':  lambda p: {p + '123', p.capitalize() + '123'},
        'suffix_1':    lambda p: {p + '1', p.capitalize() + '1'},
        'suffix_!':    lambda p: {p + '!', p.capitalize() + '!'},
        'suffix_year': lambda p: {p + '2024', p + '2023', p + '2022'},
        'prefix_123':  lambda p: {'123' + p},
        'double':      lambda p: {p + p} if len(p) <= 8 else set(),
    }

    results = {}
    for name, fn in rules.items():
        gen = {p for base in sample_base for p in fn(base) if is_valid(p)}
        hits = gen & eval_set
        results[name] = (len(hits), len(gen))
        print(f"      {name:15s}: {len(hits):5,} hits / {len(gen):8,} generes "
              f"({len(hits)/max(len(eval_set),1)*100:.3f}% coverage)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Rules Engine — augmentation de passwords')
    parser.add_argument('--input',    default=None)
    parser.add_argument('--eval',     default=os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_eval.txt'))
    parser.add_argument('--output',   default=os.path.join(BASE_DIR, 'output', 'generated', 'rules_augmented.txt'))
    parser.add_argument('--markov',   default=None)
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--no-save',  action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("RULES ENGINE — Augmentation par mutations")
    print("=" * 60)

    eval_pwds = []
    if os.path.exists(args.eval):
        with open(args.eval, 'r', encoding='utf-8', errors='ignore') as f:
            eval_pwds = [l.strip() for l in f if l.strip()]
        print(f"\nEval: {len(eval_pwds):,} passwords ({args.eval})")
    else:
        print(f"\nWarning: fichier eval non trouve ({args.eval})")

    base_pwds = []
    if args.input and os.path.exists(args.input):
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    base_pwds.append(line)
        print(f"Transformer: {len(base_pwds):,} passwords ({args.input})")
    else:
        print(f"Pas de fichier input specifie. Mode demo sur eval.")
        base_pwds = eval_pwds[:10_000] if eval_pwds else []

    if args.markov and os.path.exists(args.markov):
        with open(args.markov, 'r', encoding='utf-8', errors='ignore') as f:
            markov_pwds = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"Markov: {len(markov_pwds):,} passwords ({args.markov})")
        base_pwds = list(set(base_pwds) | set(markov_pwds))
        print(f"Combined base: {len(base_pwds):,} passwords")

    if eval_pwds:
        evaluate_coverage(set(base_pwds), eval_pwds, "AVANT regles")

    if args.analysis and eval_pwds:
        coverage_by_rule(base_pwds, eval_pwds)

    print(f"\nApplication des regles sur {len(base_pwds):,} passwords...")
    eval_set = set(eval_pwds) if eval_pwds else None

    if eval_set:
        matches, total_candidates = apply_rules(base_pwds, verbose=True, eval_set=eval_set)
        pct_after = len(matches) / len(eval_set) * 100
        print(f"\n   [APRES regles] Coverage: {len(matches):,}/{len(eval_set):,} = {pct_after:.2f}%")
        if matches:
            examples = random.sample(sorted(matches), min(10, len(matches)))
            print(f"   Exemples: {examples}")
    else:
        augmented = apply_rules(base_pwds, verbose=True)
        matches = set()
        total_candidates = len(augmented)
        pct_after = 0.0

    if not args.no_save and matches:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"# MATCHES: {len(matches):,}/{len(eval_pwds):,} = {pct_after:.2f}%\n")
            for p in sorted(matches):
                f.write(p + '\n')
        print(f"\n   Sauvegarde: {args.output}")

    print(f"\n{'='*60}")
    print(f"RESUME")
    print(f"   Base:      {len(base_pwds):,} passwords")
    print(f"   Augmente:  {total_candidates:,} candidats")
    print(f"   Facteur:   x{total_candidates/max(len(base_pwds),1):.1f}")
    if eval_pwds:
        print(f"   Coverage:  {pct_after:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
