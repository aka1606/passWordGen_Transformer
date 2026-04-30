"""
Rules Engine — Augmentation de passwords generes par mutations.

Usage:
    python scripts/analysis/rules_engine.py --input output/generated/v6_generated.txt
    python scripts/analysis/rules_engine.py --input output/generated/v6_generated.txt --eval data/splits/rockyou_eval.txt
"""

import os, sys, argparse, time, random
from itertools import product

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# REGLES
# ============================================================

LEET_MAP = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0',
    's': '$', 't': '7', 'b': '6', 'g': '9', 'l': '1',
}

LEET_MAP_LIGHT = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$',
}

SUFFIXES = [
    # Chiffres courts
    '1', '2', '3', '12', '21', '123', '1234', '12345', '123456',
    '0', '01', '007', '11', '22', '33', '00', '99', '69', '88',
    # Annees
    '2024', '2023', '2022', '2021', '2020', '2019', '2018',
    '24', '23', '22', '21', '20', '19', '18',
    '1990', '1991', '1992', '1993', '1994', '1995',
    '1996', '1997', '1998', '1999', '2000', '2001',
    # Symboles
    '!', '!!', '!!!', '!1', '1!', '!123', '123!',
    '.', '*', '#', '@', '?', '_', '-',
    # Combos frequents
    '1!', '1!!', '123!', '!@#', '1@3',
]

PREFIXES = [
    '1', '12', '123', '0', '00',
    'the', 'my', 'mr', 'ms',
]

MIN_LEN = 4
MAX_LEN = 30


def is_valid(pwd):
    return MIN_LEN <= len(pwd) <= MAX_LEN and pwd.isprintable()


def leet(pwd, leet_map):
    result = []
    for c in pwd:
        result.append(leet_map.get(c.lower(), c))
    return ''.join(result)


def generate_variants(pwd):
    """Genere toutes les variantes d'un password."""
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
    """
    Applique toutes les mutations sur base_passwords.
    Si eval_set fourni: mode streaming (RAM constante), retourne matches uniquement.
    Sinon: retourne le set complet (attention RAM).
    """
    t0 = time.time()
    total = len(base_passwords)
    total_candidates = 0

    if verbose:
        print(f"   Base: {total:,} passwords")
        print(f"   Application des regles...")

    if eval_set is not None:
        # Mode streaming: RAM constante, on check contre eval a la volee
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
        elapsed = time.time() - t0
        if verbose:
            print(f"   Done en {elapsed:.0f}s → {total_candidates:,} candidats, {len(matches):,} matches")
        return matches, total_candidates
    else:
        # Mode classique: accumule tout (attention RAM pour gros volumes)
        candidates = set()
        for i, pwd in enumerate(base_passwords):
            for v in generate_variants(pwd):
                candidates.add(v)
            if verbose and (i + 1) % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"   {i+1:,}/{total:,} ({(i+1)/total*100:.0f}%) | "
                      f"{len(candidates):,} candidats | {elapsed:.0f}s")
        elapsed = time.time() - t0
        if verbose:
            print(f"   Done en {elapsed:.0f}s → {len(candidates):,} candidats uniques")
        return candidates


# ============================================================
# EVALUATION
# ============================================================

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
    """Montre la contribution de chaque type de regle."""
    eval_set = set(eval_passwords)
    base_set = set(base_passwords)

    print("\n   Contribution par regle (sample):")
    sample_base = list(base_set)[:sample]

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
        gen = set()
        for p in sample_base:
            gen.update(fn(p))
        gen = {p for p in gen if is_valid(p)}
        hits = gen & eval_set
        results[name] = (len(hits), len(gen))
        print(f"      {name:15s}: {len(hits):5,} hits / {len(gen):8,} generes "
              f"({len(hits)/max(len(eval_set),1)*100:.3f}% coverage)")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Rules Engine — augmentation de passwords')
    parser.add_argument('--input',  default=None,
                        help='Fichier de passwords generes (un par ligne)')
    parser.add_argument('--eval',   default=os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_eval.txt'),
                        help='Fichier eval pour mesurer la couverture')
    parser.add_argument('--output', default=os.path.join(BASE_DIR, 'output', 'generated', 'rules_augmented.txt'),
                        help='Fichier de sortie')
    parser.add_argument('--markov', default=None,
                        help='Fichier Markov a combiner (optionnel)')
    parser.add_argument('--analysis', action='store_true',
                        help='Affiche la contribution par regle')
    parser.add_argument('--no-save', action='store_true',
                        help='Ne pas sauvegarder le fichier de sortie')
    args = parser.parse_args()

    print("=" * 60)
    print("RULES ENGINE — Augmentation par mutations")
    print("=" * 60)

    # Charger eval
    eval_pwds = []
    if os.path.exists(args.eval):
        with open(args.eval, 'r', encoding='utf-8', errors='ignore') as f:
            eval_pwds = [l.strip() for l in f if l.strip()]
        print(f"\nEval: {len(eval_pwds):,} passwords ({args.eval})")
    else:
        print(f"\nWarning: fichier eval non trouve ({args.eval})")

    # Charger passwords de base
    base_pwds = []

    if args.input and os.path.exists(args.input):
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    base_pwds.append(line)
        print(f"Transformer: {len(base_pwds):,} passwords ({args.input})")
    else:
        # Pas de fichier input — utilise l'eval comme demo
        print(f"Pas de fichier input specifie. Mode demo sur eval.")
        base_pwds = eval_pwds[:10_000] if eval_pwds else []

    # Charger Markov si fourni
    if args.markov and os.path.exists(args.markov):
        with open(args.markov, 'r', encoding='utf-8', errors='ignore') as f:
            markov_pwds = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"Markov: {len(markov_pwds):,} passwords ({args.markov})")
        base_pwds = list(set(base_pwds) | set(markov_pwds))
        print(f"Combined base: {len(base_pwds):,} passwords")

    # Coverage avant regles
    if eval_pwds:
        evaluate_coverage(set(base_pwds), eval_pwds, "AVANT regles")

    # Analyse par regle
    if args.analysis and eval_pwds:
        coverage_by_rule(base_pwds, eval_pwds)

    # Appliquer les regles (mode streaming si eval disponible)
    print(f"\nApplication des regles sur {len(base_pwds):,} passwords...")
    eval_set = set(eval_pwds) if eval_pwds else None

    if eval_set:
        # Mode streaming: RAM constante, seuls les matches sont gardés
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

    # Sauvegarder (uniquement les matches si mode streaming)
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
