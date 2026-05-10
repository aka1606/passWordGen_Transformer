"""
Coverage curve — évalue le taux de couverture à différentes échelles de candidats.

Usage:
    python3 scripts/analysis/coverage_curve.py \
        --generated output/generated/v7_generated.txt \
        --eval data/splits/rockyou_eval.txt [--rules]
"""

import os, argparse, random, json, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEET_MAP_LIGHT = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$'}
SUFFIXES_COMMON = ['1', '12', '123', '1234', '!', '2024', '2023', '2022', '2021', '00', '99']
MIN_LEN, MAX_LEN = 4, 30


def _leet(pwd, m):
    return ''.join(m.get(c.lower(), c) for c in pwd)


def _variants(pwd):
    if not pwd or not (MIN_LEN <= len(pwd) <= MAX_LEN and pwd.isprintable()):
        return []
    cap = pwd.capitalize()
    low = pwd.lower()
    v = {pwd, cap, low, pwd.upper(),
         _leet(pwd, LEET_MAP_LIGHT), _leet(cap, LEET_MAP_LIGHT)}
    for s in SUFFIXES_COMMON:
        v.update([pwd + s, cap + s, low + s])
    return [x for x in v if MIN_LEN <= len(x) <= MAX_LEN]


def apply_rules(passwords):
    result = set()
    for pwd in passwords:
        for v in _variants(pwd):
            result.add(v)
    return result


def load_file(path):
    passwords = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if pwd and not pwd.startswith('#'):
                passwords.append(pwd)
    return passwords


def coverage(candidates_set, eval_set):
    matches = candidates_set & eval_set
    pct = len(matches) / len(eval_set) * 100 if eval_set else 0
    return len(matches), round(pct, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', default=os.path.join(BASE_DIR, 'output', 'generated', 'generated_passwords.txt'))
    parser.add_argument('--eval',      default=os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_eval.txt'))
    parser.add_argument('--rules',     action='store_true')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--output',    default=os.path.join(BASE_DIR, 'output', 'results', 'coverage_curve.json'))
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("COVERAGE CURVE - Analyse multi-echelle")
    print("=" * 60)

    generated = load_file(args.generated)
    eval_pwds = load_file(args.eval)
    eval_set  = set(eval_pwds)
    total_gen = len(generated)

    print(f"\n  Generes disponibles : {total_gen:,}")
    print(f"  Eval (cibles)       : {len(eval_set):,}")

    thresholds = [1_000, 5_000, 10_000, 50_000, 100_000,
                  500_000, 1_000_000, 5_000_000, 10_000_000]
    thresholds = [t for t in thresholds if t <= total_gen]
    thresholds.append(total_gen)  # toujours tester le max disponible
    thresholds = sorted(set(thresholds))

    results = []

    print(f"\n{'Candidats':>12}  {'Matches':>8}  {'Coverage':>8}  {'+ Rules':>8}  {'+ Rules cand':>14}")
    print("-" * 60)

    for n in thresholds:
        subset = generated if n >= total_gen else random.sample(generated, n)
        subset_set = set(subset)
        matches, pct = coverage(subset_set, eval_set)

        row = {
            'n_candidates': n,
            'n_unique': len(subset_set),
            'matches': matches,
            'coverage_pct': pct,
        }

        if args.rules:
            t0 = time.time()
            augmented = apply_rules(subset_set)
            elapsed = time.time() - t0
            m_rules, pct_rules = coverage(augmented, eval_set)
            row['n_after_rules'] = len(augmented)
            row['matches_after_rules'] = m_rules
            row['coverage_after_rules_pct'] = pct_rules
            row['rules_time_sec'] = round(elapsed, 2)
            print(f"{n:>12,}  {matches:>8,}  {pct:>7.3f}%  {pct_rules:>7.3f}%  {len(augmented):>14,}")
        else:
            print(f"{n:>12,}  {matches:>8,}  {pct:>7.3f}%")

        results.append(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            'eval_size': len(eval_set),
            'total_generated': total_gen,
            'curve': results
        }, f, indent=2)
    print(f"\nResultats sauvegardes : {args.output}")

    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)
    for r in results:
        n = r['n_candidates']
        pct = r['coverage_pct']
        after = f"  =>  {r['coverage_after_rules_pct']:.3f}% apres rules ({r['n_after_rules']:,} cand.)" if 'coverage_after_rules_pct' in r else ""
        print(f"  {n:>10,} candidats  =>  {pct:.3f}% coverage{after}")


if __name__ == '__main__':
    main()
