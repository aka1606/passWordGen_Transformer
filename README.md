# Projet: Générateur Intelligent de Mots de Passe

## 🏗️ Structure Propre

```
/ams-projet2/
├── README.md                    # Documentation (ce fichier)
│
├── 📂 data/                     # Données brutes
│   ├── vocabulaire.txt          # 375k mots de passe uniques
│   └── TrainEval/
│       ├── train.txt            # 375k mots de passe (entraînement)
│       └── eval.txt             # 2k mots de passe (test)
│
├── 📂 scripts/                  # Scripts d'analyse
│   ├── vocab.py                 # Analyse EDA complète
│   ├── analyse_eval.py          # Comparaison train vs eval
│   ├── analyse_avancee.py       # Patterns, ngrams, positions
│   └── benchmark.py             # Benchmark RAM/capacités
│
├── 📂 analysis/                 # Documentation
│   └── strategie_analyses.md    # Plan complet du générateur
│
└── 📂 output/                   # Résultats des analyses (JSON)
    ├── stats_analysis.json
    ├── patterns_analysis.json
    └── benchmark_results.json
```

---

## 📊 Scripts Disponibles

### 1️⃣ Analyse EDA
**File:** `scripts/vocab.py` → `output/stats_analysis.json`
- Statistiques, composition, bigrammes, entropie

### 2️⃣ Comparaison Train/Eval
**File:** `scripts/analyse_eval.py`
- Différences, force des mots de passe, overlap (0%)

### 3️⃣ Patterns Avancés
**File:** `scripts/analyse_avancee.py` → `output/patterns_analysis.json`
- 4-grams, 5-grams, positions, séquences, années

### 4️⃣ Benchmark
**File:** `scripts/benchmark.py` → `output/benchmark_results.json`
- Mémoire, temps d'entraînement, verdict

---

## 🎯 Insights Clés

**Composition (375k mots):**
- 47.67%: Minuscules + Chiffres
- 33.13%: Minuscules seules
- 13.98%: Chiffres seuls

**Positionnement des Chiffres ⭐:**
- **41.80%** finissent par chiffre (RÈGLE MAJEURE)
- 18.53%: commencent par chiffre

**Éléments Courants:**
- Top 4-grams: 'love' (3007x), '1234' (2623x)
- Top 5-grams: '12345' (1035x), 'ilove' (656x)
- Années: 2000, 2007, 2008 (majoritaires)

---

## 📈 Stats

| Métrique | Valeur |
|----------|--------|
| Total | 375,853 |
| Uniques | 375,819 |
| Longueur moy | 7.60 |
| Entropie | 5.11 bits |
| Caractères uniques | 92 |

---

## 💻 Capacités du Mac

**RAM disponible:** 9.8 GB

| Modèle | Memory | Temps | ✅ Status |
|--------|--------|-------|-----------|
| Petit | 0.095 GB | 1h40 | OK |
| Moyen | 0.196 GB | 4h50 | OK |
| Grand | 0.425 GB | 13h | OK |

---

## 📚 Plus d'info

Voir [analysis/strategie_analyses.md](analysis/strategie_analyses.md)

