
import os
import sys
import json
import time
import math
import random
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

SCRIPT_VERSION = 5  # v5 Phase 2: boost LR + warm restarts + gen rapide

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'splits', 'train.txt')
    EVAL_PATH = os.path.join(BASE_DIR, 'data', 'splits', 'eval.txt')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'output', 'results', 'transformer_results.json')
    MODEL_PATH = os.path.join(BASE_DIR, 'output', 'models', 'transformer_model.pt')
    CHECKPOINT_PATH = os.path.join(BASE_DIR, 'output', 'models', 'transformer_checkpoint.pt')
    
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Modèle — taille adaptée au dataset (375k samples)
    MAX_SEQ_LEN = 32
    EMBED_DIM = 192       # adapté au dataset
    NUM_HEADS = 6         # 192/6 = 32 head_dim
    NUM_LAYERS = 6        # suffisant pour char-level
    FF_DIM = 512          # SwiGLU intermédiaire
    DROPOUT = 0.08        # ↓↓ Phase 2: pas d'overfit → libérer capacité
    
    # Entraînement — PHASE 2 BOOST (reprise depuis checkpoint v5)
    BATCH_SIZE = 512
    GRAD_ACCUM_STEPS = 1
    LEARNING_RATE = 4e-4   # ↑↑ LR restart pour sortir du plateau
    WEIGHT_DECAY = 0.005   # ↓ moins de régularisation
    EPOCHS = 100           # ↑ marge pour warm restarts
    WARMUP_STEPS = 50      # ↓↓ court (modèle déjà pré-entraîné)
    GRAD_CLIP = 0.8        # ↑ moins restrictif
    LABEL_SMOOTHING = 0.03 # ↓ moins de bruit
    USE_AMP = False        # DÉSACTIVÉ — MPS float16 instable
    
    # Cosine Warm Restarts
    LR_CYCLE_LENGTH = 2000  # steps par cycle
    LR_CYCLE_DECAY = 0.85   # decay du LR max entre cycles
    
    # Early Stopping
    VAL_SPLIT = 0.05
    PATIENCE = 12          # ↑ plus de patience pour les restarts
    MIN_DELTA = 0.001
    
    # Génération — OPTIMISÉE VITESSE
    NUM_GENERATE = 500_000  # ↑ plus de candidats
    TEMPERATURE = [0.7, 0.8, 0.9, 1.0]  # ↑ skip 0.5 (trop de duplicates)
    TOP_K = 50
    TOP_P = 0.92
    GEN_BATCH = 2048       # ↑↑ 4x plus rapide
    GEN_STALE_LIMIT = 5    # stop si N batches sans progrès

# ============================================================
# TOKENIZER
# ============================================================

class CharTokenizer:
    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    
    def __init__(self):
        self.special_tokens = [self.PAD, self.SOS, self.EOS]
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
    
    def fit(self, passwords):
        chars = sorted(set(c for pwd in passwords for c in pwd))
        self.char2idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        for i, c in enumerate(chars):
            self.char2idx[c] = len(self.special_tokens) + i
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"   Vocabulaire: {self.vocab_size} tokens ({len(chars)} chars + {len(self.special_tokens)} spéciaux)")
        return self
    
    def from_dict(self, char2idx):
        self.char2idx = char2idx
        self.idx2char = {v: k for k, v in char2idx.items()}
        self.vocab_size = len(char2idx)
        return self
    
    def encode(self, password, max_len=None):
        max_len = max_len or Config.MAX_SEQ_LEN
        tokens = [self.char2idx[self.SOS]]
        for c in password:
            if c in self.char2idx:
                tokens.append(self.char2idx[c])
        tokens.append(self.char2idx[self.EOS])
        if len(tokens) < max_len:
            tokens += [self.char2idx[self.PAD]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len - 1] + [self.char2idx[self.EOS]]
        return tokens
    
    def decode(self, indices):
        chars = []
        eos_idx = self.char2idx[self.EOS]
        pad_idx = self.char2idx[self.PAD]
        sos_idx = self.char2idx[self.SOS]
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == eos_idx:
                break
            if idx != pad_idx and idx != sos_idx:
                tok = self.idx2char.get(idx, '')
                if tok:
                    chars.append(tok)
        return ''.join(chars)
    
    def encode_all(self, passwords, max_len=None):
        """Encode tout en un seul gros tensor (perf)"""
        max_len = max_len or Config.MAX_SEQ_LEN
        encoded = []
        for pwd in passwords:
            if len(pwd) < max_len - 2:
                encoded.append(self.encode(pwd, max_len))
        return torch.tensor(encoded, dtype=torch.long)

# ============================================================
# DATASET ET DATALOADER
# ============================================================

class PasswordDataset(Dataset):
    def __init__(self, tensor_data):
        """tensor_data: (N, seq_len) long tensor, déjà encodé"""
        self.inputs = tensor_data[:, :-1].contiguous()
        self.targets = tensor_data[:, 1:].contiguous()
    
    def __len__(self):
        return self.inputs.size(0)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class RMSNorm(nn.Module):
    """Plus rapide et stable que LayerNorm (pas de mean centering)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward (meilleur que GELU, utilisé dans Llama/GPT-4)"""
    def __init__(self, dim, ff_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(dim, ff_dim, bias=False)
        self.up = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

class CausalSelfAttention(nn.Module):
   
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = dropout
    
    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, nh, T, hd)
        q, k, v = qkv.unbind(0)
        
        # KV-cache: concatène les K/V passés
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        
        new_kv = (k, v) if use_cache else None
        
        # is_causal uniquement en training (séquence complète)
        # Pendant la génération avec cache: 1 token query → pas besoin de mask
        is_causal = (past_kv is None) and (T > 1)
        dp = self.attn_dropout if self.training else 0.0
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out), new_kv

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x, offset=0):
        """offset: position de départ (0 en training, step_num avec KV-cache)"""
        return x + self.pe[:, offset:offset + x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, layer_idx=0, num_layers=1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ff = SwiGLUFFN(embed_dim, ff_dim, dropout)
    
    def forward(self, x, past_kv=None, use_cache=False):
        h = self.norm1(x)
        attn_out, new_kv = self.attn(h, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_kv

class PasswordTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_seq_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, i, num_layers)
            for i in range(num_layers)
        ])
        self.ln_final = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying
        self._init_weights()
    
    def _init_weights(self):
        """Init GPT-2 style: output projections scaled par 1/sqrt(2*n_layers)"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            # Scale les projections de sortie pour stabiliser les résidus profonds
            if 'proj.weight' in name or 'down.weight' in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
    
    def forward(self, x, past_kvs=None, use_cache=False):
        B, T = x.shape
        # Offset pour positional encoding avec KV-cache
        offset = 0 if past_kvs is None else past_kvs[0][0].shape[2]
        
        h = self.token_emb(x) * math.sqrt(self.embed_dim)
        h = self.pos_enc(h, offset=offset)
        h = self.drop(h)
        
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            h, new_kv = block(h, past_kv=past_kv, use_cache=use_cache)
            new_kvs.append(new_kv)
        
        logits = self.head(self.ln_final(h))
        
        if use_cache:
            return logits, new_kvs
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_dropout(self, p):
        """Ajuste dynamiquement le dropout (utile pour Phase 2 boost)"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p
        for block in self.blocks:
            block.attn.attn_dropout = p

# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
    
    def state_dict(self):
        return {'counter': self.counter, 'best_loss': self.best_loss}
    
    def load_state_dict(self, d):
        self.counter = d['counter']
        self.best_loss = d['best_loss']
    
    def status(self):
        return f"patience {self.counter}/{self.patience} (best={self.best_loss:.4f})"

# ============================================================
# GÉNÉRATION AVEC KV-CACHE (~5x plus rapide)
# ============================================================

@torch.no_grad()
def generate_passwords(model, tokenizer, num_generate, temperature=0.8,
                       top_k=50, top_p=0.92, device='mps', batch_size=2048,
                       stale_limit=5):
    """Génération avec KV-cache + stale detection pour éviter les doublons infinis"""
    model.eval()
    generated = set()
    sos_idx = tokenizer.char2idx[CharTokenizer.SOS]
    eos_idx = tokenizer.char2idx[CharTokenizer.EOS]
    pad_idx = tokenizer.char2idx[CharTokenizer.PAD]
    
    total_attempts = 0
    max_attempts = num_generate * 3
    stale_count = 0
    t_start = time.time()
    last_report = 0
    
    while len(generated) < num_generate and total_attempts < max_attempts:
        remaining = num_generate - len(generated)
        bs = min(batch_size, max(remaining * 2, batch_size))  # toujours au moins batch_size
        prev_size = len(generated)
        
        # Premier token: SOS
        cur_input = torch.full((bs, 1), sos_idx, dtype=torch.long, device=device)
        all_tokens = cur_input.clone()
        finished = torch.zeros(bs, dtype=torch.bool, device=device)
        past_kvs = None
        
        for step in range(Config.MAX_SEQ_LEN - 1):
            logits, past_kvs = model(cur_input, past_kvs=past_kvs, use_cache=True)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
                sorted_logits[mask] = float('-inf')
                next_logits.scatter_(1, sorted_indices, sorted_logits)
            
            next_logits[:, pad_idx] = float('-inf')
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            finished |= (next_token.squeeze(-1) == eos_idx)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
            cur_input = next_token
            
            if finished.all():
                break
        
        # Decode batch (vectorisé: EOS masking sur GPU puis decode CPU)
        eos_mask = (all_tokens == eos_idx)
        for i in range(bs):
            eos_pos = eos_mask[i].nonzero(as_tuple=False)
            end = eos_pos[0].item() if len(eos_pos) > 0 else all_tokens.size(1)
            if end > 1:  # au moins 1 char (skip SOS)
                pwd = tokenizer.decode(all_tokens[i, :end])
                if len(pwd) >= 2:
                    generated.add(pwd)
        
        total_attempts += bs
        new_added = len(generated) - prev_size
        
        # Stale detection: si trop de duplicates, abandonner cette température
        if new_added < bs * 0.005:  # moins de 0.5% de nouveaux
            stale_count += 1
            if stale_count >= stale_limit:
                elapsed = time.time() - t_start
                print(f"      ⚡ Stale détecté ({stale_count}x): {len(generated):,} uniques en {elapsed:.0f}s, passage au suivant")
                break
        else:
            stale_count = 0
        
        # Progress report tous les 10k
        if len(generated) - last_report >= 10_000:
            elapsed = time.time() - t_start
            rate = len(generated) / elapsed if elapsed > 0 else 0
            dup_rate = (1 - len(generated) / max(total_attempts, 1)) * 100
            print(f"      Générés: {len(generated):,}/{num_generate:,} | "
                  f"{rate:.0f} pwd/s | dup: {dup_rate:.0f}% | {elapsed:.0f}s")
            last_report = len(generated)
    
    elapsed = time.time() - t_start
    print(f"      Total: {len(generated):,} uniques en {elapsed:.0f}s ({len(generated)/max(elapsed,1):.0f} pwd/s)")
    return generated

# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_val(model, val_loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * x.size(0)
        mask = (y != 0)
        total_correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
        total_tokens += mask.sum().item()
    return total_loss / len(val_loader.dataset), total_correct / max(total_tokens, 1) * 100

# ============================================================
# CHECKPOINT (SAVE / LOAD)
# ============================================================

def save_checkpoint(path, model, optimizer, tokenizer, epoch, global_step,
                    best_val_loss, early_stopping, history, config):
    torch.save({
        'version': SCRIPT_VERSION,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'tokenizer_char2idx': tokenizer.char2idx,
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'early_stopping': early_stopping.state_dict(),
        'history': history,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'embed_dim': config.EMBED_DIM,
            'num_heads': config.NUM_HEADS,
            'num_layers': config.NUM_LAYERS,
            'ff_dim': config.FF_DIM,
            'max_seq_len': config.MAX_SEQ_LEN,
            'dropout': config.DROPOUT
        }
    }, path)

def load_checkpoint(path, model, optimizer, early_stopping, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Vérifier la compatibilité
    ckpt_version = ckpt.get('version', 3)
    if ckpt_version != SCRIPT_VERSION:
        print(f"   ⚠️  Checkpoint v{ckpt_version} incompatible avec script v{SCRIPT_VERSION}")
        return None
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    # Forcer tous les tensors de l'optimizer sur le bon device (fix mps/cpu mismatch)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    early_stopping.load_state_dict(ckpt['early_stopping'])
    return ckpt['epoch'], ckpt['global_step'], ckpt['best_val_loss'], ckpt['history']

# ============================================================
# TRAINING (optimisé)
# ============================================================

def get_lr(step, warmup_steps, max_lr, total_steps,
           cycle_length=2000, cycle_decay=0.85):
    """Cosine warm restarts: LR remonte périodiquement pour échapper aux plateaux"""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    effective_step = step - warmup_steps
    cycle_num = effective_step // cycle_length
    step_in_cycle = effective_step % cycle_length
    cycle_max_lr = max_lr * (cycle_decay ** cycle_num)
    min_lr = cycle_max_lr * 0.01  # floor à 1% du cycle max
    return min_lr + 0.5 * (cycle_max_lr - min_lr) * (1 + math.cos(math.pi * step_in_cycle / cycle_length))

_interrupted = False

def train_model(model, train_loader, val_loader, eval_passwords, tokenizer, config,
                start_epoch=0, global_step=0, best_val_loss=float('inf'),
                history=None, optimizer=None, early_stopping=None):
    global _interrupted
    device = config.DEVICE
    model = model.to(device)
    use_amp = config.USE_AMP and device == 'mps'
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.95))
    
    pad_idx = tokenizer.char2idx[CharTokenizer.PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=config.LABEL_SMOOTHING)
    
    if early_stopping is None:
        early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)
    
    # Total steps = optimizer steps (avec grad accum)
    steps_per_epoch = len(train_loader) // config.GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * config.EPOCHS
    if history is None:
        history = []
    accum = config.GRAD_ACCUM_STEPS
    
    def signal_handler(sig, frame):
        global _interrupted
        _interrupted = True
        print("\n\n   ⏸️  Ctrl+C détecté! Sauvegarde du checkpoint en cours...")
    
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT {'(REPRISE)' if start_epoch > 0 else ''}")
    print(f"{'='*60}")
    print(f"   Device: {device} | AMP float16: {'ON' if use_amp else 'OFF'}")
    print(f"   Paramètres: {model.count_parameters():,} ({model.count_parameters()*4/1024/1024:.1f} MB)")
    print(f"   Epochs: {start_epoch+1} → {config.EPOCHS} max")
    print(f"   Early stopping: {early_stopping.status()}")
    print(f"   Batch: {config.BATCH_SIZE} x {accum} accum = {config.BATCH_SIZE * accum} effective")
    print(f"   Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"   Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    for epoch in range(start_epoch, config.EPOCHS):
        if _interrupted:
            break
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if _interrupted:
                break
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Cosine LR schedule (sur optimizer steps)
            opt_step = global_step // accum
            lr = get_lr(opt_step, config.WARMUP_STEPS, config.LEARNING_RATE, total_steps,
                       config.LR_CYCLE_LENGTH, config.LR_CYCLE_DECAY)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward avec mixed precision
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
                loss = loss / accum  # Scale pour gradient accumulation
            
            loss.backward()
            
            with torch.no_grad():
                mask = (y != pad_idx)
                train_correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
                train_total += mask.sum().item()
            
            # Optimizer step toutes les `accum` batches
            if (batch_idx + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accum  # Unscale pour logging
            global_step += 1
            
            if batch_idx % 100 == 0:
                acc = train_correct / max(train_total, 1) * 100
                print(f"   Epoch {epoch+1}/{config.EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item()*accum:.4f} | Acc: {acc:.1f}% | LR: {lr:.2e}")
        
        if _interrupted:
            print(f"\n   ⏸️  Sauvegarde checkpoint epoch {epoch+1}...")
            save_checkpoint(config.CHECKPOINT_PATH, model, optimizer, tokenizer,
                          epoch, global_step, best_val_loss, early_stopping, history, config)
            print(f"   ✅ Checkpoint sauvegardé! Relance avec: python scripts/transformer_gen.py")
            break
        
        # ---- VALIDATION ----
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / max(train_total, 1) * 100
        val_loss, val_acc = evaluate_val(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - epoch_start
        
        overfit_gap = val_loss - avg_train_loss
        overfit_status = "⚠️  OVERFIT" if overfit_gap > 0.3 else "✅ OK"
        
        print(f"\n   ╔══════════════════════════════════════════════════╗")
        print(f"   ║ Epoch {epoch+1:2d}/{config.EPOCHS} | Time: {elapsed:.0f}s")
        print(f"   ║ Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"   ║ Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"   ║ Gap: {overfit_gap:.4f} → {overfit_status}")
        print(f"   ║ Early Stop: {early_stopping.status()}")
        print(f"   ╚══════════════════════════════════════════════════╝")
        
        # Quick coverage test toutes les 5 epochs
        quick_coverage = 0
        if (epoch + 1) % 5 == 0:
            print(f"   → Quick eval (10k)...")
            qg = generate_passwords(model, tokenizer, 10000, 0.8, 50, 0.92, device, 1024)
            es = set(eval_passwords)
            qm = qg & es
            quick_coverage = len(qm) / len(es) * 100
            print(f"   → Coverage: {len(qm)}/{len(es)} = {quick_coverage:.2f}%")
            if qm:
                print(f"   → Matchés: {list(qm)[:10]}")
        
        epoch_info = {
            'epoch': epoch + 1, 'train_loss': round(avg_train_loss, 4),
            'train_acc': round(train_acc, 2), 'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 2), 'overfit_gap': round(overfit_gap, 4),
            'lr': round(lr, 8), 'time_sec': round(elapsed, 1),
            'quick_coverage': round(quick_coverage, 2)
        }
        history.append(epoch_info)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'version': SCRIPT_VERSION,
                'model_state': model.state_dict(),
                'tokenizer_char2idx': tokenizer.char2idx,
                'epoch': epoch + 1, 'val_loss': val_loss, 'val_acc': val_acc,
                'config': {
                    'vocab_size': tokenizer.vocab_size, 'embed_dim': config.EMBED_DIM,
                    'num_heads': config.NUM_HEADS, 'num_layers': config.NUM_LAYERS,
                    'ff_dim': config.FF_DIM, 'max_seq_len': config.MAX_SEQ_LEN,
                    'dropout': config.DROPOUT
                }
            }, config.MODEL_PATH)
            print(f"   💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        
        # Save checkpoint (pour resume)
        save_checkpoint(config.CHECKPOINT_PATH, model, optimizer, tokenizer,
                       epoch + 1, global_step, best_val_loss, early_stopping, history, config)
        print(f"   💾 Checkpoint epoch {epoch+1} sauvegardé")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n   🛑 EARLY STOPPING à epoch {epoch+1}!")
            print(f"   Val loss n'a pas amélioré depuis {config.PATIENCE} epochs.")
            break
    
    signal.signal(signal.SIGINT, old_handler)
    return history, optimizer, early_stopping

# ============================================================
# EVALUATION COMPLÈTE
# ============================================================

def full_evaluation(model, tokenizer, eval_passwords, train_passwords, config):
    device = config.DEVICE
    model.eval()
    eval_set = set(eval_passwords)
    results = {}
    all_generated = set()
    
    print(f"\n{'='*60}")
    print(f"ÉVALUATION COMPLÈTE")
    print(f"{'='*60}")
    
    for temp in config.TEMPERATURE:
        print(f"\n🌡️  Temp={temp} | top_k={config.TOP_K} | top_p={config.TOP_P}")
        print(f"   Génération de {config.NUM_GENERATE:,} candidats...")
        
        t0 = time.time()
        generated = generate_passwords(model, tokenizer, config.NUM_GENERATE,
                                       temp, config.TOP_K, config.TOP_P, device, config.GEN_BATCH,
                                       config.GEN_STALE_LIMIT)
        gen_time = time.time() - t0
        all_generated |= generated
        
        matches = generated & eval_set
        coverage = len(matches) / len(eval_set) * 100
        train_set = set(train_passwords)
        overlap = generated & train_set
        lengths = [len(p) for p in generated]
        
        result = {
            'temperature': temp,
            'total_generated': len(generated),
            'generation_time_sec': round(gen_time, 1),
            'coverage': {'matches': len(matches), 'total_eval': len(eval_set), 'pct': round(coverage, 4)},
            'overlap_train': {'count': len(overlap), 'pct': round(len(overlap)/max(len(generated),1)*100, 2)},
            'avg_length': round(sum(lengths)/max(len(lengths),1), 2),
            'matched_examples': sorted(list(matches))[:30],
            'sample_generated': random.sample(list(generated), min(20, len(generated)))
        }
        results[f'temp_{temp}'] = result
        
        print(f"   Générés: {len(generated):,} en {gen_time:.1f}s")
        print(f"   📊 Coverage: {len(matches)}/{len(eval_set)} = {coverage:.2f}%")
        print(f"   Overlap train: {len(overlap)} ({result['overlap_train']['pct']}%)")
        if matches:
            print(f"   Matchés: {sorted(list(matches))[:15]}")
    
    all_matches = all_generated & eval_set
    combined_coverage = len(all_matches) / len(eval_set) * 100
    
    print(f"\n{'='*60}")
    print(f"📊 COVERAGE COMBINÉ")
    print(f"{'='*60}")
    print(f"   Total généré: {len(all_generated):,}")
    print(f"   Matches: {len(all_matches)}/{len(eval_set)} = {combined_coverage:.2f}%")
    if all_matches:
        print(f"   Exemples: {sorted(list(all_matches))[:50]}")
    
    # Sauvegarder matches + tous les générés dans un seul fichier
    gen_path = os.path.join(config.BASE_DIR, 'output', 'generated', 'generated_passwords.txt')
    with open(gen_path, 'w', encoding='utf-8') as f:
        f.write(f"# === MATCHES ({len(all_matches)}/{len(eval_set)} = {combined_coverage:.2f}% coverage) ===\n")
        f.write(f"# Mots de passe de eval retrouvés par le modèle\n")
        f.write("#\n")
        for pwd in sorted(all_matches):
            f.write(pwd + '\n')
        f.write(f"\n# === TOUS LES GÉNÉRÉS ({len(all_generated):,} uniques) ===\n")
        f.write(f"# Ensemble complet des candidats générés\n")
        f.write("#\n")
        for pwd in sorted(all_generated):
            f.write(pwd + '\n')
    print(f"\n💾 {len(all_matches)} matches + {len(all_generated):,} générés → {gen_path}")
    
    results['combined'] = {
        'total_unique_generated': len(all_generated),
        'coverage_matches': len(all_matches),
        'coverage_pct': round(combined_coverage, 4),
        'all_matched': sorted(list(all_matches))
    }
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
    
    config = Config()
    reset = '--reset' in sys.argv
    generate_only = '--generate-only' in sys.argv
    quick_mode = '--quick' in sys.argv
    
    if quick_mode:
        config.NUM_GENERATE = 10_000
        config.TEMPERATURE = [0.8]
        generate_only = True  # quick implique generate-only
        print("⚡ MODE QUICK: 10k candidats, T=0.8 uniquement")
    
    print("=" * 60)
    print("🤖 TRANSFORMER v5 Phase 2 — BOOST + GÉNÉRATION RAPIDE")
    print(f"   Device: {config.DEVICE.upper()} | AMP={config.USE_AMP} | Ctrl+C safe")
    print(f"   Archi: {config.NUM_LAYERS}L-{config.NUM_HEADS}H-{config.EMBED_DIM}D | SwiGLU+RMSNorm")
    print(f"   LR: {config.LEARNING_RATE} | Warm restarts: cycle={config.LR_CYCLE_LENGTH}, decay={config.LR_CYCLE_DECAY}")
    print("=" * 60)
    
    # Charger données
    print("\n📂 Chargement des données...")
    def load_passwords(path):
        pwds = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                pwd = line.strip().rstrip('\\')
                if pwd and 2 <= len(pwd) <= config.MAX_SEQ_LEN - 2:
                    pwds.append(pwd)
        return pwds
    
    all_train_pwds = load_passwords(config.TRAIN_PATH)
    eval_pwds = load_passwords(config.EVAL_PATH)
    
    # Train/Val split (deterministic)
    random.shuffle(all_train_pwds)
    val_size = int(len(all_train_pwds) * config.VAL_SPLIT)
    val_pwds = all_train_pwds[:val_size]
    train_pwds = all_train_pwds[val_size:]
    
    print(f"   Train: {len(train_pwds):,} | Val: {len(val_pwds):,} | Eval: {len(eval_pwds):,}")
    
    # Check for existing checkpoint
    has_checkpoint = os.path.exists(config.CHECKPOINT_PATH) and not reset
    
    if reset:
        for f in [config.CHECKPOINT_PATH, config.MODEL_PATH]:
            if os.path.exists(f):
                os.remove(f)
                print(f"   🗑️  Supprimé: {os.path.basename(f)}")
        has_checkpoint = False
    
    if has_checkpoint:
        # Vérifier compatibilité
        print(f"\n📥 Checkpoint trouvé, vérification...")
        ckpt_preview = torch.load(config.CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        ckpt_version = ckpt_preview.get('version', 3)
        
        if ckpt_version != SCRIPT_VERSION:
            print(f"   ⚠️  Checkpoint v{ckpt_version} incompatible avec v{SCRIPT_VERSION}")
            print(f"   → Suppression et restart depuis zéro")
            os.remove(config.CHECKPOINT_PATH)
            if os.path.exists(config.MODEL_PATH):
                os.remove(config.MODEL_PATH)
            has_checkpoint = False
            del ckpt_preview
        else:
            del ckpt_preview
            print(f"   ✅ Compatible (v{SCRIPT_VERSION})")
    
    if has_checkpoint:
        # RESUME from checkpoint
        print(f"   Reprise de l'entraînement...")
        ckpt = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)
        
        tokenizer = CharTokenizer().from_dict(ckpt['tokenizer_char2idx'])
        print(f"   Tokenizer restauré: {tokenizer.vocab_size} tokens")
        
        model = PasswordTransformer(
            tokenizer.vocab_size, config.EMBED_DIM, config.NUM_HEADS,
            config.NUM_LAYERS, config.FF_DIM, config.MAX_SEQ_LEN, config.DROPOUT
        ).to(config.DEVICE)  # MPS AVANT optimizer pour éviter device mismatch
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.95))
        early_stopping = EarlyStopping(config.PATIENCE, config.MIN_DELTA)
        
        result = load_checkpoint(config.CHECKPOINT_PATH, model, optimizer, early_stopping, config.DEVICE)
        if result is None:
            print("   ❌ Checkpoint invalide, restart depuis zéro")
            has_checkpoint = False
        else:
            start_epoch, global_step, best_val_loss, history = result
            
            # Phase 2: appliquer nouveau dropout dynamiquement
            model.set_dropout(config.DROPOUT)
            print(f"   Reprise à epoch {start_epoch+1}, step {global_step}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            print(f"   Early stop: {early_stopping.status()}")
            print(f"   Dropout ajusté: {config.DROPOUT} (Phase 2)")
            print(f"   Historique: {len(history)} epochs faits")
    
    if not has_checkpoint:
        # FRESH start
        print("\n🔤 Construction du tokenizer...")
        tokenizer = CharTokenizer()
        tokenizer.fit(all_train_pwds + eval_pwds)
        
        model = PasswordTransformer(
            tokenizer.vocab_size, config.EMBED_DIM, config.NUM_HEADS,
            config.NUM_LAYERS, config.FF_DIM, config.MAX_SEQ_LEN, config.DROPOUT
        )
        print(f"   Paramètres: {model.count_parameters():,} ({model.count_parameters()*4/1024/1024:.1f} MB)")
        
        optimizer = None
        early_stopping = None
        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')
        history = []
    
    # DataLoaders avec dataset pré-tensorisé
    print("\n📦 Pré-tensorisation des datasets...")
    t_enc = time.time()
    train_tensor = tokenizer.encode_all(train_pwds)
    val_tensor = tokenizer.encode_all(val_pwds)
    print(f"   Encodé en {time.time()-t_enc:.1f}s | Train: {train_tensor.shape} | Val: {val_tensor.shape}")
    
    train_dataset = PasswordDataset(train_tensor)
    val_dataset = PasswordDataset(val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=0)
    
    if not generate_only:
        history, optimizer, early_stopping = train_model(
            model, train_loader, val_loader, eval_pwds, tokenizer, config,
            start_epoch, global_step, best_val_loss, history, optimizer, early_stopping
        )
    
    # Charger le meilleur modèle pour la génération
    if os.path.exists(config.MODEL_PATH):
        print("\n📥 Chargement du meilleur modèle...")
        best_ckpt = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt['model_state'])
        model = model.to(config.DEVICE)
        print(f"   Epoch {best_ckpt['epoch']}, val_loss={best_ckpt['val_loss']:.4f}")
    
    # Évaluation complète
    eval_results = full_evaluation(model, tokenizer, eval_pwds, all_train_pwds, config)
    
    # Sauvegarder (pas en mode quick pour ne pas écraser les vrais résultats)
    if not quick_mode:
        final_results = {
            'version': SCRIPT_VERSION,
            'config': {
                'architecture': 'SwiGLU+RMSNorm+SDPA+KV-cache',
                'device': config.DEVICE, 'embed_dim': config.EMBED_DIM,
                'num_heads': config.NUM_HEADS, 'num_layers': config.NUM_LAYERS,
                'ff_dim': config.FF_DIM, 'max_seq_len': config.MAX_SEQ_LEN,
                'batch_size': config.BATCH_SIZE, 'grad_accum': config.GRAD_ACCUM_STEPS,
                'epochs_run': len(history),
                'lr': config.LEARNING_RATE, 'patience': config.PATIENCE,
                'label_smoothing': config.LABEL_SMOOTHING,
                'val_split': config.VAL_SPLIT, 'params': model.count_parameters()
            },
            'training_history': history,
            'evaluation': eval_results
        }
        
        with open(config.OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    else:
        print("\n⚡ Mode quick: transformer_results.json NON écrasé")
    
    combined = eval_results.get('combined', {})
    print(f"\n{'='*60}")
    print(f"🏆 RÉSUMÉ FINAL")
    print(f"{'='*60}")
    print(f"   Epochs: {len(history)}")
    print(f"   Coverage: {combined.get('coverage_pct', 0):.2f}%")
    print(f"   Matches: {combined.get('coverage_matches', 0)}/{len(eval_pwds)}")
    print(f"   Total généré: {combined.get('total_unique_generated', 0):,}")
    print(f"\n💾 Résultats: {config.OUTPUT_PATH}")
    print(f"💾 Modèle: {config.MODEL_PATH}")

if __name__ == '__main__':
    main()
