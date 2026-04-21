# Stratégie pour Générateur de Mots de Passe Intelligent

## 1. **PATTERNS STRUCTURELS** 🏗️ (PRIORITÉ 1)
```
Analyser les patterns qui répètent:
- [Mot dict] + [Chiffres]  (ex: password123)
- [Minuscules] + [Majuscule début] + [Chiffres]  (ex: Admin2021)
- [Mots multiple] concat  (ex: correcthorsebattery)
- Substitutions leet (ex: P@ssw0rd - a->@, o->0, l->1)
- Patterns keyboard (ex: qwerty, asdfgh, 1234)
```
→ **Résultat:** Rules engine pour générer des variantes

---

## 2. **WORDLIST & DICTIONNAIRE** 📚 (PRIORITÉ 1)
```
Extraire les mots réels du vocabulaire (vocabulaire.txt):
- Mots de passe pur texte (sans chiffres ni spéciaux)
- Fréquence des mots complets
- Utiliser pour base + variantes
```
→ **Résultat:** Base de mots + transformations

---

## 3. **CHAÎNES MARKOV** 🔗 (PRIORITÉ 2)
```
Modèle probabiliste de transitions:
- Bigrammes/Trigrammes (déjà en partie fait)
- Matrice de probabilité: char[i] → char[i+1]
- Générer des séquences réalistes
```
→ **Résultat:** Générateur pseudo-aléatoire réaliste

---

## 4. **POSITIONS DE CARACTÈRES** 📍 (PRIORITÉ 2)
```
Où apparaissent les caractères spéciaux/chiffres?
- Au début/fin (très courant)
- Entre 2 groupes de lettres
- À des positions spécifiques
```
→ **Résultat:** Heuristiques positionnelles

---

## 5. **SOUS-CHAÎNES COURANTES** 🔍 (PRIORITÉ 2)
```
Extraire les 4-5 char substrings courants:
- "1234", "123", "000", "999"
- "pass", "admin", "test", "user"
- Bigrammes qui répètent
```
→ **Résultat:** Fragments à combiner intelligemment

---

## 6. **TRANSFORMATIONS COURANTES** 🔄 (PRIORITÉ 3)
```
- Capitalisation patterns (1ère lettre, 1ère + dernière)
- Leet speak courant (3->E, 4->A, 0->O, 1->I/L)
- Replacements simples (@, !, $)
- Années (1990-2025)
- Doublons de caractères (ll, ss, ee)
```
→ **Résultat:** Règles de mutation

---

## 7. **SÉQUENCES NUMÉRIQUES** 🔢 (PRIORITÉ 2)
```
- Années populaires (1990-2024)
- Séquences (123, 456, 789, 1234, 9999)
- Doublons (11, 22, 33, ... 99)
- Compter la fréquence
```
→ **Résultat:** Injection ciblée de chiffres

---

## 8. **ENTROPIE PAR CLASSE** 📊 (PRIORITÉ 3)
```
Analyser separately:
- Mots entièrement minuscules
- Alphanumérique simple
- Avec majuscules
- Avec spéciaux
Générer selon la distribution observée
```
→ **Résultat:** Stratégie de classe appropriée

---

## 9. **CLUSTERISATION DE PATTERNS** 🎯 (PRIORITÉ 3)
```
Grouper par pattern similaire:
- Groupe "noms + années"
- Groupe "clavier"
- Groupe "noms communs"
- Groupe "aléatoire complet"
→ Générer par cluster avec poids probabiliste
```

---

## 10. **ANALYSE NGRAM AVANCÉE** 🔀 (PRIORITÉ 1)
```
Au-delà des bigrammes/trigrammes:
- 4-grams, 5-grams courants
- Modèle probabiliste complet
- Markov chain d'ordre 3-4
```
→ **Résultat:** Générateur contextuel puissant

---

## ORDRE D'IMPLÉMENTATION RECOMMANDÉ:

### Phase 1 (Analyses fondamentales):
1. ✅ Analyse basique (FAIT)
2. Extraire wordlist propre du vocabulaire
3. Ngrams avancés (4-5 grams)
4. Patterns structurels courants

### Phase 2 (Modèle intelligent):
5. Markov chains
6. Positions de caractères spéciaux/chiffres
7. Transformations courantes

### Phase 3 (Générateur):
8. Builder qui combine tout
9. Validation contre données réelles
10. Optimisation par coverage

---

## MÉTRIQUE DE SUCCÈS:
- **Coverage:** % de mots de passe réels du test qu'on peut générer/matcher
- **Diversité:** Nombre unique de candidats générés
- **Similarité:** À quel point les générés ressemblent aux réels
