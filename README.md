# Generateur de mots de passe par Transformer

Systeme de generation de mots de passe base sur un Transformer decodeur (GPT-style)
entraine sur RockYou et sources externes, augmente par un moteur de regles de mutation.

---

## Structure du projet

```
ams-projet2/
├── README.md
├── data/
│   ├── raw/                          # rockyou.txt brut (gitignored)
│   ├── extra/                        # Sources additionnelles (gitignored)
│   └── splits/
│       ├── rockyou_eval.txt          # 143k mots de passe d'evaluation
│       ├── rockyou_train.txt         # 14M mots de passe (gitignored)
│       └── combined_train.txt        # 25M freq-weighted (gitignored)
│
├── scripts/
│   ├── preprocess/
│   │   ├── download_rockyou.py       # Telecharge RockYou depuis SourceForge
│   │   ├── download_extra.py         # Telecharge SecLists (Pwdb, 000webhost, phpbb)
│   │   └── prepare_dataset.py        # Fusionne et prepare combined_train.txt
│   │
│   ├── training/
│   │   ├── train_server_v7.py        # v7: 10.4M parametres, RoPE, 25M freq-weighted
│   │
│   ├── analysis/
│   │   ├── rules_engine.py           # Moteur de mutation (capitalize, leet, suffixes)
│   │   └── coverage_curve.py          # Mesure coverage vs taille du dico
│   │
│   ├── slurm/                        # Jobs SLURM pour cluster
│   │   ├── job_v7.sh                 # Entrainement v7
│   │   └── generate_job_v7.sh        # Generation v7
│   │
│   └── utils/
│       ├── setup_env.sh              # Setup conda env
│       └── setup_server.sh           # Setup cluster
│
└── output/
    ├── models/                       # Checkpoints (.pt, gitignored)
    ├── generated/                    # Mots de passe generes (gitignored)
    └── results/                      # Logs SLURM et JSON (.log gitignored)
```

---

## Pipeline complet

### 1. Preparation des donnees (sur le cluster)

```bash
# Telecharger RockYou
python3 scripts/preprocess/download_rockyou.py

# Telecharger les sources additionnelles (SecLists)
python3 scripts/preprocess/download_extra.py

# Fusionner avec freq-weighting et filtre ASCII
python3 scripts/preprocess/prepare_dataset.py --freq-weight --max-repeat 5
# Sortie: data/splits/combined_train.txt (~25M mots de passe)
```

### 2. Entrainement (cluster SLURM)

```bash
sbatch scripts/slurm/job_v7.sh        # Job 24h, a relancer pour continuer
```

### 3. Generation et evaluation

```bash
sbatch scripts/slurm/generate_job_v7.sh
# Sortie: output/generated/v7_generated.txt
# Sortie: output/results/v7_results.json
```

### 4. Augmentation par regles

```bash
python3 scripts/analysis/rules_engine.py \
    --input output/generated/v7_generated.txt \
    --eval data/splits/rockyou_eval.txt
# 1M candidats genere environ 134M apres mutations
```

---

## Architecture v7

| Parametre | Valeur |
|-----------|--------|
| Couches | 12 |
| Dimension embedding | 256 |
| Tetes d'attention | 8 |
| Dimension FFN | 768 |
| Position encoding | RoPE (Rotary) |
| Total parametres | 10.4M |
| Vocabulaire | char-level (207 tokens) |
| Longueur max sequence | 32 |

Composants techniques : RMSNorm, SwiGLU, RoPE, SDPA causal, KV-cache, AMP float16.

---

## Dataset combine (25M mots de passe)

| Source | Taille | Note |
|--------|--------|------|
| RockYou avec frequences | 14.3M | freq-weighted, max_repeat=5 |
| Pwdb top 10M | 10M | compilation multi-breach |
| 000webhost | 720k | leak service d'hebergement |
| phpbb | 184k | leak forum communautaire |

Filtre ASCII printable (32-126) applique pour garder un vocabulaire compact.

---

## Hyperparametres d'entrainement

| Parametre | Valeur |
|-----------|--------|
| Batch size | 1024 |
| Gradient accumulation | 4 (batch effectif = 4096) |
| Learning rate initial | 3e-4 |
| Schedule LR | Cosine warm restarts |
| Optimizer | AdamW (beta1=0.9, beta2=0.95) |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Label smoothing | 0.05 |
| Mixed precision | float16 (AMP) |
| Early stopping patience | 12 epochs |

---

## Resultats v6 (reference)

| Metrique | Valeur |
|----------|--------|
| Val loss (epoch 25) | 2.7983 |
| Val accuracy | 31.2% |
| Coverage brut (1M candidats) | 1.84% |
| Coverage avec rules engine (134M candidats) | 3.52% |

Les resultats v7 seront ajoutes dans le rapport apres convergence.

---

## Rapport

Le rapport (PDF/LaTeX) n'est pas inclus dans ce dépôt tel qu'il est actuellement. Les métriques et résultats intermédiaires sont dans output/results/.
