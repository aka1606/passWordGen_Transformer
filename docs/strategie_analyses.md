# Strategie pour generateur de mots de passe intelligent

## 1. Patterns structurels (priorite 1)

```
Analyser les patterns recurrents :
- [Mot dictionnaire] + [Chiffres]  (ex: password123)
- [Minuscules] + [Majuscule debut] + [Chiffres]  (ex: Admin2021)
- [Mots multiples] concatenes  (ex: correcthorsebattery)
- Substitutions leet  (ex: P@ssw0rd : a -> @, o -> 0, l -> 1)
- Patterns clavier  (ex: qwerty, asdfgh, 1234)
```

Resultat : moteur de regles pour generer des variantes.

---

## 2. Wordlist et dictionnaire (priorite 1)

```
Extraire les mots reels du vocabulaire :
- Mots de passe en pur texte (sans chiffres ni speciaux)
- Frequence des mots complets
- Utiliser comme base pour les variantes
```

Resultat : base de mots et transformations associees.

---

## 3. Chaines de Markov (priorite 2)

```
Modele probabiliste de transitions :
- Bigrammes et trigrammes (deja partiellement fait)
- Matrice de probabilite : char[i] -> char[i+1]
- Generer des sequences realistes
```

Resultat : generateur pseudo-aleatoire realiste.

---

## 4. Positions de caracteres (priorite 2)

```
Analyser ou apparaissent les caracteres speciaux et chiffres :
- Au debut ou en fin (tres courant)
- Entre deux groupes de lettres
- A des positions specifiques
```

Resultat : heuristiques positionnelles.

---

## 5. Sous-chaines courantes (priorite 2)

```
Extraire les substrings de 4-5 caracteres frequents :
- "1234", "123", "000", "999"
- "pass", "admin", "test", "user"
- Bigrammes recurrents
```

Resultat : fragments a combiner intelligemment.

---

## 6. Transformations courantes (priorite 3)

```
- Patterns de capitalisation (1ere lettre, 1ere et derniere)
- Leet speak courant (3 -> E, 4 -> A, 0 -> O, 1 -> I/L)
- Substitutions simples (@, !, $)
- Annees (1990-2025)
- Doublons de caracteres (ll, ss, ee)
```

Resultat : regles de mutation.

---

## 7. Sequences numeriques (priorite 2)

```
- Annees populaires (1990-2024)
- Sequences (123, 456, 789, 1234, 9999)
- Doublons (11, 22, 33, ... 99)
- Comptage des frequences
```

Resultat : injection ciblee de chiffres.

---

## 8. Entropie par classe (priorite 3)

```
Analyser separement :
- Mots entierement minuscules
- Alphanumerique simple
- Avec majuscules
- Avec speciaux

Generer selon la distribution observee.
```

Resultat : strategie adaptee a chaque classe.

---

## 9. Clustering de patterns (priorite 3)

```
Grouper par pattern similaire :
- Groupe "noms + annees"
- Groupe "clavier"
- Groupe "noms communs"
- Groupe "aleatoire complet"

Generer par cluster avec poids probabiliste.
```

---

## 10. Analyse n-gram avancee (priorite 1)

```
Au-dela des bigrammes et trigrammes :
- 4-grams, 5-grams courants
- Modele probabiliste complet
- Markov chain d'ordre 3 ou 4
```

Resultat : generateur contextuel puissant.

---

## Ordre d'implementation recommande

### Phase 1 (analyses fondamentales)
1. Analyse basique (fait)
2. Extraire wordlist propre du vocabulaire
3. N-grams avances (4-5 grams)
4. Patterns structurels courants

### Phase 2 (modele intelligent)
5. Chaines de Markov
6. Positions des caracteres speciaux et chiffres
7. Transformations courantes

### Phase 3 (generateur)
8. Builder qui combine toutes les approches
9. Validation contre donnees reelles
10. Optimisation par coverage

---

## Metriques de succes

- Coverage : pourcentage de mots de passe reels du test que l'on peut generer ou matcher.
- Diversite : nombre unique de candidats generes.
- Similarite : ressemblance entre les generes et les reels.
