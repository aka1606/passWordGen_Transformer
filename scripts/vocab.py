import re
from collections import Counter
import json
import numpy as np

def analyze_passwords(fichier):
    """Analyse complète des mots de passe"""
    
    passwords = []
    lengths = []
    
    with open(fichier, 'r', encoding='utf-8') as f:
        for ligne in f:
            pwd = ligne.strip()
            if pwd:
                passwords.append(pwd)
                lengths.append(len(pwd))
    
    print("="*60)
    print("ANALYSE EDA COMPLÈTE DES MOTS DE PASSE")
    print("="*60)
    
    # 1. STATISTIQUES DE BASE
    print("\n1. STATISTIQUES DE BASE")
    print(f"   Total de mots de passe: {len(passwords)}")
    print(f"   Mots de passe uniques: {len(set(passwords))}")
    print(f"   Doublons: {len(passwords) - len(set(passwords))}")
    
    # 2. LONGUEURS
    print("\n2. ANALYSE DES LONGUEURS")
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = np.mean(lengths)
    median_len = np.median(lengths)
    std_len = np.std(lengths)
    
    print(f"   Longueur min: {min_len}")
    print(f"   Longueur max: {max_len}")
    print(f"   Longueur moyenne: {avg_len:.2f}")
    print(f"   Longueur médiane: {median_len:.0f}")
    print(f"   Écart-type: {std_len:.2f}")
    
    # Distribution des longueurs
    length_dist = Counter(lengths)
    print(f"\n   Top 10 longueurs les plus fréquentes:")
    for length, count in sorted(length_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / len(passwords)) * 100
        print(f"      Longueur {length}: {count} ({pct:.2f}%)")
    
    # 3. COMPOSITION DES CARACTÈRES
    print("\n3. COMPOSITION DES CARACTÈRES")
    
    lowercase_count = 0
    uppercase_count = 0
    digit_count = 0
    special_count = 0
    space_count = 0
    
    passwords_with_lower = 0
    passwords_with_upper = 0
    passwords_with_digit = 0
    passwords_with_special = 0
    
    for pwd in passwords:
        has_lower = False
        has_upper = False
        has_digit = False
        has_special = False
        
        for char in pwd:
            if char.islower():
                lowercase_count += 1
                has_lower = True
            elif char.isupper():
                uppercase_count += 1
                has_upper = True
            elif char.isdigit():
                digit_count += 1
                has_digit = True
            elif char == ' ':
                space_count += 1
            else:
                special_count += 1
                has_special = True
        
        if has_lower:
            passwords_with_lower += 1
        if has_upper:
            passwords_with_upper += 1
        if has_digit:
            passwords_with_digit += 1
        if has_special:
            passwords_with_special += 1
    
    total_chars = lowercase_count + uppercase_count + digit_count + special_count + space_count
    
    print(f"   Caractères minuscules: {lowercase_count} ({lowercase_count/total_chars*100:.2f}%)")
    print(f"   Caractères majuscules: {uppercase_count} ({uppercase_count/total_chars*100:.2f}%)")
    print(f"   Chiffres: {digit_count} ({digit_count/total_chars*100:.2f}%)")
    print(f"   Caractères spéciaux: {special_count} ({special_count/total_chars*100:.2f}%)")
    print(f"   Espaces: {space_count} ({space_count/total_chars*100:.2f}%)")
    
    print(f"\n   Mots de passe contenant:")
    print(f"      Minuscules: {passwords_with_lower} ({passwords_with_lower/len(passwords)*100:.2f}%)")
    print(f"      Majuscules: {passwords_with_upper} ({passwords_with_upper/len(passwords)*100:.2f}%)")
    print(f"      Chiffres: {passwords_with_digit} ({passwords_with_digit/len(passwords)*100:.2f}%)")
    print(f"      Caractères spéciaux: {passwords_with_special} ({passwords_with_special/len(passwords)*100:.2f}%)")
    
    # 4. PATTERNS COURANTS
    print("\n4. PATTERNS COURANTS")
    
    starts_with_upper = sum(1 for p in passwords if p and p[0].isupper())
    starts_with_lower = sum(1 for p in passwords if p and p[0].islower())
    starts_with_digit = sum(1 for p in passwords if p and p[0].isdigit())
    
    ends_with_digit = sum(1 for p in passwords if p and p[-1].isdigit())
    ends_with_special = sum(1 for p in passwords if p and p[-1] in '!@#$%^&*()_+-=[]{}|;:,.<>?')
    
    print(f"   Commence par majuscule: {starts_with_upper} ({starts_with_upper/len(passwords)*100:.2f}%)")
    print(f"   Commence par minuscule: {starts_with_lower} ({starts_with_lower/len(passwords)*100:.2f}%)")
    print(f"   Commence par chiffre: {starts_with_digit} ({starts_with_digit/len(passwords)*100:.2f}%)")
    print(f"\n   Se termine par chiffre: {ends_with_digit} ({ends_with_digit/len(passwords)*100:.2f}%)")
    print(f"   Se termine par caractère spécial: {ends_with_special} ({ends_with_special/len(passwords)*100:.2f}%)")
    
    # 5. BIGRAMMES (2 caractères consécutifs)
    print("\n5. TOP 20 BIGRAMMES LES PLUS FRÉQUENTS")
    
    bigrammes = Counter()
    for pwd in passwords:
        for i in range(len(pwd) - 1):
            bigrammes[pwd[i:i+2]] += 1
    
    for bigram, count in bigrammes.most_common(20):
        pct = (count / sum(bigrammes.values())) * 100
        print(f"   '{bigram}': {count} ({pct:.2f}%)")
    
    # 6. TRIGRAMMES (3 caractères consécutifs)
    print("\n6. TOP 10 TRIGRAMMES LES PLUS FRÉQUENTS")
    
    trigrammes = Counter()
    for pwd in passwords:
        for i in range(len(pwd) - 2):
            trigrammes[pwd[i:i+3]] += 1
    
    for trigram, count in trigrammes.most_common(10):
        pct = (count / sum(trigrammes.values())) * 100
        print(f"   '{trigram}': {count} ({pct:.2f}%)")
    
    # 7. CATÉGORIES DE MOTS DE PASSE
    print("\n7. CATÉGORIES DE MOTS DE PASSE")
    
    only_lower = sum(1 for p in passwords if p.islower() and p.isalpha())
    only_upper = sum(1 for p in passwords if p.isupper() and p.isalpha())
    only_digits = sum(1 for p in passwords if p.isdigit())
    alpha_numeric = sum(1 for p in passwords if any(c.isalpha() for c in p) and any(c.isdigit() for c in p) and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in p))
    with_special = sum(1 for p in passwords if any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in p))
    
    print(f"   Seulement minuscules: {only_lower} ({only_lower/len(passwords)*100:.2f}%)")
    print(f"   Seulement majuscules: {only_upper} ({only_upper/len(passwords)*100:.2f}%)")
    print(f"   Seulement chiffres: {only_digits} ({only_digits/len(passwords)*100:.2f}%)")
    print(f"   Alphanumérique: {alpha_numeric} ({alpha_numeric/len(passwords)*100:.2f}%)")
    print(f"   Contient caractères spéciaux: {with_special} ({with_special/len(passwords)*100:.2f}%)")
    
    # 8. ENTROPIE
    print("\n8. ENTROPIE")
    
    all_chars = Counter()
    for pwd in passwords:
        for char in pwd:
            all_chars[char] += 1
    
    total = sum(all_chars.values())
    entropy = -sum((count/total) * np.log2(count/total) for count in all_chars.values())
    
    print(f"   Entropie Shannon: {entropy:.2f} bits")
    print(f"   Nombre de caractères uniques: {len(all_chars)}")
    
    # 9. EXEMPLES
    print("\n9. EXEMPLES DE MOTS DE PASSE")
    print(f"   Plus court: {min(passwords, key=len)}")
    print(f"   Plus long: {max(passwords, key=len)}")
    print(f"   Premier 10:")
    for i, pwd in enumerate(passwords[:10], 1):
        print(f"      {i}. {pwd}")
    
    # 10. SAUVEGARDE DES STATISTIQUES
    stats = {
        'total': len(passwords),
        'uniques': len(set(passwords)),
        'longueur_min': int(min_len),
        'longueur_max': int(max_len),
        'longueur_moyenne': float(avg_len),
        'longueur_mediane': float(median_len),
        'longueur_std': float(std_len),
        'entropie': float(entropy),
        'caracteres_uniques': len(all_chars),
        'avec_minuscules': passwords_with_lower,
        'avec_majuscules': passwords_with_upper,
        'avec_chiffres': passwords_with_digit,
        'avec_speciaux': passwords_with_special,
    }
    
    with open('stats_analysis.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Statistiques sauvegardées dans 'stats_analysis.json'")
    print("="*60)
    
    return passwords, stats

# Exécution
if __name__ == "__main__":
    passwords, stats = analyze_passwords('TrainEval/train.txt')
