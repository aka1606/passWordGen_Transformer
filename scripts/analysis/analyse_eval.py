#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des données d'évaluation et entraînement
"""

import json
from collections import Counter
import re
import math

def analyze_passwords(file_path):
    """Analyse un fichier de mots de passe"""
    passwords = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pwd = line.strip()
            if pwd:
                passwords.append(pwd)
    
    return passwords

def get_password_characteristics(pwd):
    """Extrait les caractéristiques d'un mot de passe"""
    characteristics = {
        'length': len(pwd),
        'has_lower': bool(re.search(r'[a-z]', pwd)),
        'has_upper': bool(re.search(r'[A-Z]', pwd)),
        'has_digit': bool(re.search(r'\d', pwd)),
        'has_special': bool(re.search(r'[^a-zA-Z0-9]', pwd)),
        'starts_with_digit': pwd[0].isdigit() if pwd else False,
        'ends_with_digit': pwd[-1].isdigit() if pwd else False,
        'consecutive_digits': len(re.findall(r'\d{2,}', pwd)) > 0,
    }
    return characteristics

def password_strength(pwd):
    """Calcule un score de force du mot de passe (0-4)"""
    score = 0
    if len(pwd) >= 8:
        score += 1
    if re.search(r'[a-z]', pwd):
        score += 1
    if re.search(r'[A-Z]', pwd):
        score += 1
    if re.search(r'\d', pwd):
        score += 1
    if re.search(r'[^a-zA-Z0-9]', pwd):
        score += 1
    return min(score, 4)

print("=" * 60)
print("ANALYSE COMPARATIVE TRAIN/EVAL")
print("=" * 60)

# Charger les données
print("\nChargement des données...")
import os as _os
_BASE = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
eval_passwords  = analyze_passwords(_os.path.join(_BASE, 'data', 'splits', 'eval.txt'))
train_passwords = analyze_passwords(_os.path.join(_BASE, 'data', 'splits', 'train.txt'))

print(f"Eval: {len(eval_passwords)} mots de passe")
print(f"Train: {len(train_passwords)} mots de passe")

# Analyser EVAL
print("\n" + "=" * 60)
print("1. ANALYSE DES MOTS DE PASSE EN ÉVALUATION")
print("=" * 60)

eval_lengths = [len(p) for p in eval_passwords]
print(f"\nLongueurs:")
print(f"  Min: {min(eval_lengths)}")
print(f"  Max: {max(eval_lengths)}")
print(f"  Moyenne: {sum(eval_lengths)/len(eval_lengths):.2f}")

eval_length_dist = Counter(eval_lengths)
top_eval_lengths = eval_length_dist.most_common(5)
print(f"\n  Top 5 longueurs:")
for length, count in top_eval_lengths:
    print(f"    Longueur {length}: {count} ({100*count/len(eval_passwords):.2f}%)")

# Caractéristiques EVAL
chars = [get_password_characteristics(p) for p in eval_passwords]
print(f"\nCaractéristiques:")
print(f"  Contient minuscules: {sum(1 for c in chars if c['has_lower'])} ({100*sum(1 for c in chars if c['has_lower'])/len(chars):.2f}%)")
print(f"  Contient majuscules: {sum(1 for c in chars if c['has_upper'])} ({100*sum(1 for c in chars if c['has_upper'])/len(chars):.2f}%)")
print(f"  Contient chiffres: {sum(1 for c in chars if c['has_digit'])} ({100*sum(1 for c in chars if c['has_digit'])/len(chars):.2f}%)")
print(f"  Contient spéciaux: {sum(1 for c in chars if c['has_special'])} ({100*sum(1 for c in chars if c['has_special'])/len(chars):.2f}%)")

# Force des mots de passe EVAL
strengths = [password_strength(p) for p in eval_passwords]
strength_dist = Counter(strengths)
print(f"\nForce des mots de passe (0-4):")
for strength in range(5):
    count = strength_dist.get(strength, 0)
    print(f"  Force {strength}: {count} ({100*count/len(eval_passwords):.2f}%)")

# Top patterns EVAL
print(f"\nTop 15 mots de passe en évaluation:")
pwd_counter = Counter(eval_passwords)
for i, (pwd, count) in enumerate(pwd_counter.most_common(15), 1):
    print(f"  {i:2d}. '{pwd}': {count} fois")

# Bigrammes EVAL
print(f"\nTop 10 bigrammes en évaluation:")
bigrams = []
for pwd in eval_passwords:
    for i in range(len(pwd)-1):
        bigrams.append(pwd[i:i+2])
bigram_counter = Counter(bigrams)
for i, (bg, count) in enumerate(bigram_counter.most_common(10), 1):
    print(f"  {i:2d}. '{bg}': {count} ({100*count/len(bigrams):.2f}%)")

# Analyser TRAIN
print("\n" + "=" * 60)
print("2. ANALYSE DES DONNÉES D'ENTRAÎNEMENT")
print("=" * 60)

train_lengths = [len(p) for p in train_passwords]
print(f"\nLongueurs:")
print(f"  Min: {min(train_lengths)}")
print(f"  Max: {max(train_lengths)}")
print(f"  Moyenne: {sum(train_lengths)/len(train_lengths):.2f}")

train_length_dist = Counter(train_lengths)
top_train_lengths = train_length_dist.most_common(5)
print(f"\n  Top 5 longueurs:")
for length, count in top_train_lengths:
    print(f"    Longueur {length}: {count} ({100*count/len(train_passwords):.2f}%)")

# Top patterns TRAIN
print(f"\nTop 15 mots de passe en entraînement:")
train_pwd_counter = Counter(train_passwords)
for i, (pwd, count) in enumerate(train_pwd_counter.most_common(15), 1):
    print(f"  {i:2d}. '{pwd}': {count} fois")

# Comparaison
print("\n" + "=" * 60)
print("3. COMPARAISON EVAL vs TRAIN")
print("=" * 60)

common_pwds = set(eval_passwords) & set(train_passwords)
print(f"\nMots de passe en commun: {len(common_pwds)}")

eval_set = set(eval_passwords)
train_set = set(train_passwords)
only_eval = eval_set - train_set
only_train = train_set - eval_set

print(f"Uniques à EVAL: {len(only_eval)}")
print(f"Uniques à TRAIN: {len(only_train)}")

# Overlap
overlap_percentage = (len(common_pwds) / len(eval_set) * 100) if eval_set else 0
print(f"Overlap: {overlap_percentage:.2f}%")

print("\n" + "=" * 60)
print("Analyse sauvegardée!")
print("=" * 60)
