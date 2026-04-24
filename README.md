# Générateur de Mots de Passe par Transformer

Système de génération de mots de passe basé sur un Transformer décodeur (GPT-style)
entraîné sur RockYou + sources externes, augmenté par un moteur de règles de mutation.

---

## 📁 Structure

```
ams-projet2/
├── README.md
├── data/
│   ├── raw/                          # rockyou.txt brut (gitignored)
│   ├── extra/                        # Sources additionnelles (gitignored)
│   └── splits/
│       ├── rockyou_eval.txt          # 143k passwords d'évaluation
│       ├── rockyou_train.txt         # 14M passwords (gitignored)
│       └── combined_train.txt        # 25M freq-weighted (gitignored)
│
├── scripts/
│   ├── preprocess/
│   │   ├── download_rockyou.py       # Télécharge RockYou depuis SourceForge
│   │   ├── download_extra.py         # Télécharge SecLists (Pwdb, 000webhost, phpbb)
│   │   └── prepare_dataset.py        # Fusionne et prépare combined_train.txt
│   │
│   ├── training/
│   │   ├── train_server.py           # v6: 7M params, RockYou seul
│   │   ├── train_server_v7.py        # v7: 10.7M params, RoPE, 25M freq-weighted
│   │   ├── markov_chain.py           # Modèle Markov (baseline)
│   │   └── transformer_gen.py        # Version locale (Mac)
│   │
│   ├── analysis/
│   │   ├── rules_engine.py           # Moteur de mutation (capitalize, leet, suffixes...)
│   │   ├── analyse_eval.py
│   │   ├── analyse_avancee.py
│   │   ├── pca_clustering.py
│   │   ├── benchmark.py
│   │   └── vocab.py
│   │
│   ├── slurm/                        # Jobs SLURM pour cluster
│   │   ├── job.sh                    # Entraînement v6
│   │   ├── job_v7.sh                 # Entraînement v7
│   │   ├── generate_job.sh           # Génération v6
│   │   └── generate_job_v7.sh        # Génération v7
│   │
│   └── utils/
│       ├── setup_env.sh              # Setup conda env
│       └── setup_server.sh           # Setup cluster
│
├── docs/
│   ├── rapport.tex                   # Rapport LaTeX
│   └── strategie_analyses.md
│
└── output/
    ├── models/                       # Checkpoints (.pt, gitignored)
    ├── generated/                    # Passwords générés (gitignored)
    └── results/                      # Logs SLURM + JSON (.log gitignored)
```

---

## 🚀 Pipeline

### 1. Préparation des données (sur le cluster)

```bash
# Télécharger RockYou
python scripts/preprocess/download_rockyou.py

# Télécharger les sources additionnelles (SecLists)
python scripts/preprocess/download_extra.py

# Fusionner avec freq-weighting
python scripts/preprocess/prepare_dataset.py --freq-weight --max-repeat 5
# → data/splits/combined_train.txt (25M passwords)
```

### 2. Entraînement (cluster SLURM)

```bash
sbatch scripts/slurm/job_v7.sh        # 24h job, à relancer
```

### 3. Génération + évaluation

```bash
sbatch scripts/slurm/generate_job_v7.sh
# → output/generated/v7_generated.txt
# → output/results/v7_results.json
```

### 4. Augmentation par règles

```bash
python scripts/analysis/rules_engine.py \
    --input output/generated/v7_generated.txt \
    --eval data/splits/rockyou_eval.txt
# → 1M candidats → 134M après mutations
```

---

## 🏗️ Architecture v7

| Paramètre | Valeur |
|-----------|--------|
| Couches | 12 |
| Dim embedding | 256 |
| Têtes | 8 |
| FFN dim | 768 |
| Position encoding | **RoPE** (Rotary) |
| Total paramètres | ~10.7M |
| Vocabulaire | char-level (~95 tokens) |

**Composants** : RMSNorm + SwiGLU + RoPE + SDPA causal + KV-cache + AMP float16

---

## 📊 Dataset combiné (25M passwords)

| Source | Taille | Note |
|--------|--------|------|
| RockYou (avec fréquences) | 14.3M | freq-weighted, max_repeat=5 |
| Pwdb top 10M | 10M | multi-breach compilation |
| 000webhost | 720k | leak hosting service |
| phpbb | 184k | leak forum community |

---

## 📚 Rapport

Voir [docs/rapport.tex](docs/rapport.tex) pour les détails complets et résultats.
