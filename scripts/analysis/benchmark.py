#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Vérifier les capacités du Mac pour un Transformer
"""

import psutil
import os
import json
import time

print("=" * 70)
print("BENCHMARK: CAPACITÉS DU MAC POUR TRANSFORMER")
print("=" * 70)

# 1. SPECS DU SYSTÈME
print("\n1️⃣  SPECS DU SYSTÈME")
print("-" * 70)

cpu_count = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
ram_total = psutil.virtual_memory().total / (1024**3)
ram_available = psutil.virtual_memory().available / (1024**3)
swap_total = psutil.swap_memory().total / (1024**3)

print(f"CPU (cores physiques): {cpu_count}")
print(f"CPU (threads logiques): {cpu_count_logical}")
print(f"RAM totale: {ram_total:.1f} GB")
print(f"RAM disponible: {ram_available:.1f} GB")
print(f"SWAP totale: {swap_total:.1f} GB")

# Vérifier GPU
has_gpu = False
try:
    import torch
    has_gpu = torch.backends.mps.is_available()  # Metal Performance Shaders (Apple Silicon)
    if has_gpu:
        print(f"GPU: ✅ Metal Performance Shaders (MPS) disponible")
    else:
        print(f"GPU: ❌ Pas de GPU accéléré détecté")
except:
    print(f"GPU: ⚠️ PyTorch non installé")

# 2. DATASET ANALYSIS
print("\n2️⃣  DATASET")
print("-" * 70)

train_size = 375_853
eval_size = 1_999
vocab_size = 131_457
avg_seq_length = 7.6  # from analysis

print(f"Train samples: {train_size:,}")
print(f"Eval samples: {eval_size:,}")
print(f"Total: {train_size + eval_size:,}")
print(f"Vocab size: {vocab_size:,}")
print(f"Avg sequence length: {avg_seq_length:.1f}")

# 3. TRANSFORMER ESTIMATIONS
print("\n3️⃣  TRANSFORMER ESTIMATIONS")
print("-" * 70)

# Petit modèle
embedding_dim_small = 64
layers_small = 2
heads_small = 4
ff_dim_small = 256
seq_length = 12  # Padding to 12

# Formule de paramètres pour Transformer:
# Embedding: vocab_size * embedding_dim
# Self-attention per layer: 3 * (embedding_dim * embedding_dim) + embedding_dim  (Q, K, V + output)
# Feed-forward per layer: 2 * embedding_dim * ff_dim + 2 * embedding_dim
# Normalization: 2 * embedding_dim per layer

params_small = (
    vocab_size * embedding_dim_small +  # Embedding layer
    layers_small * (
        # Self-attention
        3 * (embedding_dim_small * embedding_dim_small) + embedding_dim_small +
        # Feed-forward
        2 * embedding_dim_small * ff_dim_small + 2 * embedding_dim_small +
        # Layer norm
        2 * embedding_dim_small
    )
)

memory_small = (params_small * 4 / (1024**3))  # 4 bytes per float32

print(f"PETIT MODÈLE (2 layers, 64 dim):")
print(f"  Paramètres: {params_small:,.0f}")
print(f"  Memory (weights): {memory_small:.3f} GB")
print(f"  Memory (+ activations): ~{memory_small * 3:.3f} GB")
print(f"  Faisable: {'✅ OUI' if memory_small * 3 < ram_available else '❌ NON'}")

# Modèle moyen
embedding_dim_med = 128
layers_med = 4
heads_med = 8
ff_dim_med = 512

params_med = (
    vocab_size * embedding_dim_med +
    layers_med * (
        3 * (embedding_dim_med * embedding_dim_med) + embedding_dim_med +
        2 * embedding_dim_med * ff_dim_med + 2 * embedding_dim_med +
        2 * embedding_dim_med
    )
)

memory_med = (params_med * 4 / (1024**3))

print(f"\nMODÈLE MOYEN (4 layers, 128 dim):")
print(f"  Paramètres: {params_med:,.0f}")
print(f"  Memory (weights): {memory_med:.3f} GB")
print(f"  Memory (+ activations): ~{memory_med * 3:.3f} GB")
print(f"  Faisable: {'✅ OUI' if memory_med * 3 < ram_available else '❌ NON'}")

# Grand modèle
embedding_dim_large = 256
layers_large = 6
heads_large = 8
ff_dim_large = 1024

params_large = (
    vocab_size * embedding_dim_large +
    layers_large * (
        3 * (embedding_dim_large * embedding_dim_large) + embedding_dim_large +
        2 * embedding_dim_large * ff_dim_large + 2 * embedding_dim_large +
        2 * embedding_dim_large
    )
)

memory_large = (params_large * 4 / (1024**3))

print(f"\nGRAND MODÈLE (6 layers, 256 dim):")
print(f"  Paramètres: {params_large:,.0f}")
print(f"  Memory (weights): {memory_large:.3f} GB")
print(f"  Memory (+ activations): ~{memory_large * 3:.3f} GB")
print(f"  Faisable: {'✅ OUI' if memory_large * 3 < ram_available else '❌ NON'}")

# 4. TIME ESTIMATIONS
print("\n4️⃣  ESTIMATIONS DE TEMPS D'ENTRAÎNEMENT")
print("-" * 70)

batch_size = 32
epochs = 10
samples_per_batch = train_size / batch_size

# Time per batch (rough estimate)
# Small model: ~0.05s per batch
# Medium model: ~0.15s per batch
# Large model: ~0.4s per batch

time_small = samples_per_batch * epochs * 0.05 / 60
time_med = samples_per_batch * epochs * 0.15 / 60
time_large = samples_per_batch * epochs * 0.4 / 60

print(f"Pour {epochs} epochs, batch_size={batch_size}:")
print(f"  Petit modèle: ~{time_small:.1f} minutes")
print(f"  Modèle moyen: ~{time_med:.1f} minutes ({time_med/60:.1f} heures)")
print(f"  Grand modèle: ~{time_large:.1f} minutes ({time_large/60:.1f} heures)")

# 5. RECOMMANDATIONS
print("\n5️⃣  RECOMMANDATIONS")
print("-" * 70)

if ram_available < 8:
    print("⚠️  RAM disponible < 8GB: Impossible un Transformer")
elif ram_available < 16:
    print(f"✅ Recommandé: PETIT MODÈLE (2 layers, 64 dim)")
    print(f"   Memory: {memory_small * 3:.3f} GB / {ram_available:.1f} GB disponible")
else:
    print(f"✅ Recommandé: MODÈLE MOYEN (4 layers, 128 dim)")
    print(f"   Memory: {memory_med * 3:.3f} GB / {ram_available:.1f} GB disponible")
    if ram_available > 32:
        print(f"✅ Optionnel: GRAND MODÈLE possible (6 layers, 256 dim)")

print(f"\n📊 Utiliser MPS (GPU Apple): {'✅ Accélération ~2-3x' if has_gpu else '❌ CPU only'}")

# 6. ALTERNATIVE: MARKOV + RULES (plus rapide)
print("\n6️⃣  ALTERNATIVE À TRANSFORMER")
print("-" * 70)
print("💡 Pour ce dataset, considérer:")
print("   - Markov chains simples (~10 min d'entraînement)")
print("   - Règles probabilistes + ngrams (~5 min)")
print("   - Hybrid: Markov + RNN simple (au lieu de Transformer)")

# Sauvegarder résultats
results = {
    "system": {
        "cpu_cores": cpu_count,
        "cpu_logical": cpu_count_logical,
        "ram_total_gb": round(ram_total, 1),
        "ram_available_gb": round(ram_available, 1),
        "gpu_available": has_gpu,
    },
    "dataset": {
        "train_samples": train_size,
        "eval_samples": eval_size,
        "vocab_size": vocab_size,
        "avg_seq_length": avg_seq_length,
    },
    "models": {
        "small": {
            "params": params_small,
            "memory_gb": round(memory_small, 3),
            "total_memory_gb": round(memory_small * 3, 3),
            "feasible": memory_small * 3 < ram_available,
            "training_time_minutes": round(time_small, 1),
        },
        "medium": {
            "params": params_med,
            "memory_gb": round(memory_med, 3),
            "total_memory_gb": round(memory_med * 3, 3),
            "feasible": memory_med * 3 < ram_available,
            "training_time_minutes": round(time_med, 1),
        },
        "large": {
            "params": params_large,
            "memory_gb": round(memory_large, 3),
            "total_memory_gb": round(memory_large * 3, 3),
            "feasible": memory_large * 3 < ram_available,
            "training_time_minutes": round(time_large, 1),
        }
    },
    "recommendation": "small" if ram_available < 16 else "medium",
}

import os as _os; _base = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
with open(_os.path.join(_base, 'output', 'results', 'benchmark_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("✅ Résultats sauvegardés dans output/benchmark_results.json")
print("=" * 70)
