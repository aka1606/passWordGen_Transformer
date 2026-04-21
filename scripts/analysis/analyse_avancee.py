#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyses avancées pour générateur intelligent de mots de passe
Phase 1: Fondamentaux critiques
"""

import json, os, re
from collections import Counter
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("PHASE 1: ANALYSES FONDAMENTALES POUR GÉNÉRATEUR INTELLIGENT")
print("=" * 70)

# 1. WORDLIST PROPRE - Extraire les mots sans chiffres ni spéciaux
print("\n1️⃣  EXTRACTION WORDLIST PROPRE")
print("-" * 70)

with open(os.path.join(BASE_DIR, 'data', 'splits', 'vocabulaire.txt'), 'r', encoding='utf-8', errors='ignore') as f:
    vocab = [line.strip() for line in f if line.strip()]

# Mots purs (seulement lettres)
pure_words = [w for w in vocab if re.match(r'^[a-zA-Z]+$', w)]
print(f"Total vocabulaire: {len(vocab)}")
print(f"Mots purs (lettres seules): {len(pure_words)} ({100*len(pure_words)/len(vocab):.2f}%)")

# Mots avec chiffres
words_with_digits = [w for w in vocab if re.search(r'\d', w)]
print(f"Mots avec chiffres: {len(words_with_digits)} ({100*len(words_with_digits)/len(vocab):.2f}%)")

# Mots avec spéciaux
words_with_special = [w for w in vocab if re.search(r'[^a-zA-Z0-9]', w)]
print(f"Mots avec spéciaux: {len(words_with_special)} ({100*len(words_with_special)/len(vocab):.2f}%)")

top_pure_words = Counter(pure_words).most_common(20)
print(f"\nTop 20 mots purs (les plus fréquents):")
for i, (word, count) in enumerate(top_pure_words, 1):
    print(f"  {i:2d}. '{word}': {count} fois")

# 2. NGRAMS AVANCÉS
print("\n\n2️⃣  NGRAMS AVANCÉS (4-grams, 5-grams)")
print("-" * 70)

# Charger train.txt
with open(os.path.join(BASE_DIR, 'data', 'splits', 'train.txt'), 'r', encoding='utf-8', errors='ignore') as f:
    train = [line.strip() for line in f if line.strip()]

# 4-grams
four_grams = []
for pwd in train:
    for i in range(len(pwd)-3):
        four_grams.append(pwd[i:i+4])

four_gram_counter = Counter(four_grams)
print(f"\nTop 15 4-grams:")
for i, (fg, count) in enumerate(four_gram_counter.most_common(15), 1):
    print(f"  {i:2d}. '{fg}': {count} ({100*count/len(four_grams):.2f}%)")

# 5-grams
five_grams = []
for pwd in train:
    for i in range(len(pwd)-4):
        five_grams.append(pwd[i:i+5])

five_gram_counter = Counter(five_grams)
print(f"\nTop 15 5-grams:")
for i, (fg, count) in enumerate(five_gram_counter.most_common(15), 1):
    print(f"  {i:2d}. '{fg}': {count} ({100*count/len(five_grams):.2f}%)")

# 3. PATTERNS STRUCTURELS
print("\n\n3️⃣  PATTERNS STRUCTURELS")
print("-" * 70)

def classify_pattern(pwd):
    """Classifie le pattern d'un mot de passe"""
    has_lower = bool(re.search(r'[a-z]', pwd))
    has_upper = bool(re.search(r'[A-Z]', pwd))
    has_digit = bool(re.search(r'\d', pwd))
    has_special = bool(re.search(r'[^a-zA-Z0-9]', pwd))
    
    if has_special:
        return "avec_spéciaux"
    elif has_lower and has_upper and has_digit:
        return "AlphaNum_mixte"
    elif has_lower and has_digit:
        return "alphanum_bas_chif"
    elif has_lower and has_upper:
        return "Mixte_casse"
    elif has_digit:
        return "chiffres_seuls"
    elif has_lower:
        return "minuscules_seules"
    elif has_upper:
        return "majuscules_seules"
    else:
        return "autre"

patterns = [classify_pattern(p) for p in train]
pattern_dist = Counter(patterns)

print(f"\nDistribution des patterns:")
for pattern, count in sorted(pattern_dist.items(), key=lambda x: -x[1]):
    print(f"  {pattern:25s}: {count:7d} ({100*count/len(train):5.2f}%)")

# 4. POSITIONS DES CHIFFRES ET SPÉCIAUX
print("\n\n4️⃣  POSITIONS DES CARACTÈRES SPÉCIAUX")
print("-" * 70)

digit_positions = {"début": 0, "fin": 0, "milieu": 0, "absent": 0}
special_positions = {"début": 0, "fin": 0, "milieu": 0, "absent": 0}

for pwd in train:
    if re.search(r'\d', pwd):
        if pwd[0].isdigit():
            digit_positions["début"] += 1
        elif pwd[-1].isdigit():
            digit_positions["fin"] += 1
        else:
            digit_positions["milieu"] += 1
    else:
        digit_positions["absent"] += 1
    
    if re.search(r'[^a-zA-Z0-9]', pwd):
        if pwd[0] in r'!@#$%^&*':
            special_positions["début"] += 1
        elif pwd[-1] in r'!@#$%^&*':
            special_positions["fin"] += 1
        else:
            special_positions["milieu"] += 1
    else:
        special_positions["absent"] += 1

print(f"\nPositions des CHIFFRES:")
for pos, count in digit_positions.items():
    print(f"  {pos:10s}: {count:7d} ({100*count/len(train):5.2f}%)")

print(f"\nPositions des SPÉCIAUX:")
for pos, count in special_positions.items():
    print(f"  {pos:10s}: {count:7d} ({100*count/len(train):5.2f}%)")

# 5. SÉQUENCES NUMÉRIQUES COURANTES
print("\n\n5️⃣  SÉQUENCES NUMÉRIQUES COURANTES")
print("-" * 70)

num_sequences = []
for pwd in train:
    # Trouver toutes les séquences de chiffres
    matches = re.findall(r'\d+', pwd)
    num_sequences.extend(matches)

seq_counter = Counter(num_sequences)
print(f"\nTop 20 séquences numériques:")
for i, (seq, count) in enumerate(seq_counter.most_common(20), 1):
    print(f"  {i:2d}. '{seq}': {count:6d} ({100*count/len(num_sequences):5.2f}%)")

# Analyser les années
years = [int(s) for s in num_sequences if len(s) == 4 and 1900 <= int(s) <= 2030]
year_dist = Counter(years)
print(f"\nTop 10 années détectées:")
for i, (year, count) in enumerate(year_dist.most_common(10), 1):
    print(f"  {i:2d}. {year}: {count} fois")

# 6. TRANSFORMATIONS COURANTES (LEET SPEAK, CAPS)
print("\n\n6️⃣  TRANSFORMATIONS COURANTES")
print("-" * 70)

# Capitalisation patterns
starts_caps = sum(1 for p in train if p and p[0].isupper())
ends_caps = sum(1 for p in train if p and p[-1].isupper())
all_caps = sum(1 for p in train if p and p.isupper() and len(p) > 1)

print(f"\nCapitalisation:")
print(f"  Commence par majuscule: {starts_caps} ({100*starts_caps/len(train):.2f}%)")
print(f"  Finit par majuscule: {ends_caps} ({100*ends_caps/len(train):.2f}%)")
print(f"  Tout en majuscules: {all_caps} ({100*all_caps/len(train):.2f}%)")

# Leet speak patterns
leet_patterns = {
    '@': sum(1 for p in train if '@' in p),
    '!': sum(1 for p in train if '!' in p),
    '#': sum(1 for p in train if '#' in p),
    '$': sum(1 for p in train if '$' in p),
    '0': sum(1 for p in train if re.search(r'\b0+\b|^0+|0+$', p)),
    '1': sum(1 for p in train if re.search(r'\b1+\b|^1+|1+$', p)),
}

print(f"\nLeet speak courant:")
for char, count in sorted(leet_patterns.items(), key=lambda x: -x[1]):
    print(f"  '{char}': {count:6d} ({100*count/len(train):5.2f}%)")

# 7. SUBSTRINGS COURANTS (4-char min)
print("\n\n7️⃣  SOUS-CHAÎNES COURANTES (fragments réutilisables)")
print("-" * 70)

# Extraire tous les substrings de 4 chars
substrings = []
for pwd in train:
    if len(pwd) >= 4:
        for i in range(len(pwd)-3):
            substrings.append(pwd[i:i+4])

substring_counter = Counter(substrings)
print(f"\nTop 20 substrings 4-char (potentiellement réutilisables):")
for i, (substr, count) in enumerate(substring_counter.most_common(20), 1):
    print(f"  {i:2d}. '{substr}': {count:6d} ({100*count/len(substrings):.2f}%)")

# Sauvegarder résultats
results = {
    "pure_words_count": len(pure_words),
    "top_pure_words": [(w, c) for w, c in top_pure_words],
    "top_4grams": [(fg, c) for fg, c in four_gram_counter.most_common(30)],
    "top_5grams": [(fg, c) for fg, c in five_gram_counter.most_common(30)],
    "patterns": dict(pattern_dist),
    "digit_positions": digit_positions,
    "top_sequences": dict(seq_counter.most_common(50)),
    "top_years": dict(year_dist.most_common(20)),
}

with open(os.path.join(BASE_DIR, 'output', 'results', 'patterns_analysis.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("✅ Résultats sauvegardés dans patterns_analysis.json")
print("=" * 70)
