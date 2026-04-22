"""
PasswordTransformer v6 — Serveur CUDA (Tesla P100 16GB)
Dataset: RockYou (14M passwords)
Changements vs v5: CUDA+AMP+GradScaler, modele 3x plus grand, batch 4x plus grand
"""

import os, sys, json, time, math, random, signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

SCRIPT_VERSION = 6

# ============================================================
# CONFIG
# ============================================================

class Config:
    BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_PATH      = os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_train.txt')
    EVAL_PATH       = os.path.join(BASE_DIR, 'data', 'splits', 'rockyou_eval.txt')
    OUTPUT_PATH     = os.path.join(BASE_DIR, 'output', 'results', 'v6_results.json')
    GEN_PATH        = os.path.join(BASE_DIR, 'output', 'generated', 'v6_generated.txt')

    # Checkpoints sur /dev/shm (RAM disk) si dispo, sinon NFS
    _FAST_DIR       = '/dev/shm/pwdgen_v6' if os.path.exists('/dev/shm') else os.path.join(BASE_DIR, 'output', 'models')
    MODEL_PATH      = os.path.join(_FAST_DIR, 'v6_model.pt')
    CHECKPOINT_PATH = os.path.join(_FAST_DIR, 'v6_checkpoint.pt')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Architecture — 3x plus grand que v5 pour 38x plus de donnees
    MAX_SEQ_LEN = 32
    EMBED_DIM   = 256     # 192 → 256
    NUM_HEADS   = 8       # 6 → 8  (head_dim = 32)
    NUM_LAYERS  = 8       # 6 → 8
    FF_DIM      = 768     # 512 → 768
    DROPOUT     = 0.1

    # Entrainement
    BATCH_SIZE       = 2048    # 512 → 2048 (P100 16GB)
    GRAD_ACCUM_STEPS = 2       # effective batch = 4096
    LEARNING_RATE    = 3e-4
    WEIGHT_DECAY     = 0.01
    EPOCHS           = 50
    WARMUP_STEPS     = 500
    GRAD_CLIP        = 1.0
    LABEL_SMOOTHING  = 0.05
    USE_AMP          = True    # float16 stable sur CUDA

    # LR cosine warm restarts
    LR_CYCLE_LENGTH  = 5000
    LR_CYCLE_DECAY   = 0.80

    # Early stopping
    VAL_SPLIT  = 0.02     # 2% de 14M = ~280k val
    PATIENCE   = 8
    MIN_DELTA  = 0.0005

    # Generation
    NUM_GENERATE    = 1_000_000
    TEMPERATURE     = [0.7, 0.8, 0.9, 1.0, 1.1]
    TOP_K           = 50
    TOP_P           = 0.92
    GEN_BATCH       = 8192
    GEN_STALE_LIMIT = 5

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
        print(f"   Vocabulaire: {self.vocab_size} tokens ({len(chars)} chars + {len(self.special_tokens)} speciaux)")
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
        """Encode avec numpy pre-allocation (rapide pour 14M samples)"""
        max_len = max_len or Config.MAX_SEQ_LEN
        sos = self.char2idx[self.SOS]
        eos = self.char2idx[self.EOS]
        pad = self.char2idx[self.PAD]

        valid = [pwd for pwd in passwords if 1 <= len(pwd) <= max_len - 2]
        n = len(valid)
        print(f"   Encodage de {n:,} passwords (numpy pre-alloc)...")
        t0 = time.time()

        result = np.full((n, max_len), pad, dtype=np.int16)  # int16 = 4x moins de RAM
        for i, pwd in enumerate(valid):
            tokens = [sos]
            for c in pwd:
                if c in self.char2idx:
                    tokens.append(self.char2idx[c])
            tokens.append(eos)
            length = min(len(tokens), max_len)
            result[i, :length] = tokens[:length]

            if (i + 1) % 1_000_000 == 0:
                elapsed = time.time() - t0
                print(f"   {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%) en {elapsed:.0f}s")

        print(f"   Encode en {time.time()-t0:.1f}s → tensor {result.shape}")
        return torch.from_numpy(result)

# ============================================================
# DATASET
# ============================================================

class PasswordDataset(Dataset):
    def __init__(self, tensor_data):
        self.inputs  = tensor_data[:, :-1].contiguous()
        self.targets = tensor_data[:, 1:].contiguous()

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]  # int16, converti en long sur GPU


def gpu_batch_iter(tensor, batch_size, drop_last=True):
    """Itere sur un tensor GPU deja en VRAM — zero copie CPU->GPU par batch."""
    n = tensor.size(0)
    indices = torch.randperm(n, device=tensor.device)
    end = n - batch_size + 1 if drop_last else n
    for start in range(0, end, batch_size):
        idx = indices[start:start + batch_size]
        batch = tensor[idx]          # [B, seq_len], deja sur GPU
        yield batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

# ============================================================
# ARCHITECTURE (identique v5: RMSNorm + SwiGLU + SDPA + KV-cache)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(dim, ff_dim, bias=False)
        self.up   = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = dropout

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_kv = (k, v) if use_cache else None
        is_causal = (past_kv is None) and (T > 1)
        dp = self.attn_dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out), new_kv


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position  = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term  = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, offset=0):
        return x + self.pe[:, offset:offset + x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, num_layers=1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn  = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ff    = SwiGLUFFN(embed_dim, ff_dim, dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        h = self.norm1(x)
        attn_out, new_kv = self.attn(h, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_kv


class PasswordTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_layers = num_layers
        self.token_emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc    = PositionalEncoding(embed_dim, max_seq_len)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, num_layers)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            if 'proj.weight' in name or 'down.weight' in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))

    def forward(self, x, past_kvs=None, use_cache=False):
        B, T = x.shape
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

# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def state_dict(self):
        return {'counter': self.counter, 'best_loss': self.best_loss}

    def load_state_dict(self, d):
        self.counter   = d['counter']
        self.best_loss = d['best_loss']

    def status(self):
        return f"patience {self.counter}/{self.patience} (best={self.best_loss:.4f})"

# ============================================================
# GENERATION avec KV-cache
# ============================================================

@torch.no_grad()
def generate_passwords(model, tokenizer, num_generate, temperature=0.8,
                       top_k=50, top_p=0.92, device='cuda', batch_size=8192,
                       stale_limit=5):
    model.eval()
    generated   = set()
    sos_idx = tokenizer.char2idx[CharTokenizer.SOS]
    eos_idx = tokenizer.char2idx[CharTokenizer.EOS]
    pad_idx = tokenizer.char2idx[CharTokenizer.PAD]

    total_attempts = 0
    max_attempts   = num_generate * 4
    stale_count    = 0
    t_start        = time.time()
    last_report    = 0

    while len(generated) < num_generate and total_attempts < max_attempts:
        bs        = min(batch_size, max((num_generate - len(generated)) * 2, batch_size))
        prev_size = len(generated)

        cur_input = torch.full((bs, 1), sos_idx, dtype=torch.long, device=device)
        all_tokens = cur_input.clone()
        finished   = torch.zeros(bs, dtype=torch.bool, device=device)
        past_kvs   = None

        for step in range(Config.MAX_SEQ_LEN - 1):
            logits, past_kvs = model(cur_input, past_kvs=past_kvs, use_cache=True)
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < topk_vals[:, -1:]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
                sorted_logits[mask] = float('-inf')
                next_logits.scatter_(1, sorted_indices, sorted_logits)

            next_logits[:, pad_idx] = float('-inf')
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            finished  |= (next_token.squeeze(-1) == eos_idx)
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
            cur_input  = next_token

            if finished.all():
                break

        eos_mask = (all_tokens == eos_idx)
        for i in range(bs):
            eos_pos = eos_mask[i].nonzero(as_tuple=False)
            end = eos_pos[0].item() if len(eos_pos) > 0 else all_tokens.size(1)
            if end > 1:
                pwd = tokenizer.decode(all_tokens[i, :end])
                if len(pwd) >= 2:
                    generated.add(pwd)

        total_attempts += bs
        new_added = len(generated) - prev_size

        if new_added < bs * 0.005:
            stale_count += 1
            if stale_count >= stale_limit:
                print(f"      Stale ({stale_count}x): {len(generated):,} uniques, passage au suivant")
                break
        else:
            stale_count = 0

        if len(generated) - last_report >= 50_000:
            elapsed = time.time() - t_start
            rate    = len(generated) / elapsed if elapsed > 0 else 0
            print(f"      Generes: {len(generated):,}/{num_generate:,} | {rate:.0f} pwd/s | {elapsed:.0f}s")
            last_report = len(generated)

    elapsed = time.time() - t_start
    print(f"      Total: {len(generated):,} uniques en {elapsed:.0f}s ({len(generated)/max(elapsed,1):.0f} pwd/s)")
    return generated

# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_val(model, val_loader, criterion, device, scaler_enabled=True):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    for x, y in val_loader:
        x = x.to(device, dtype=torch.long, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=scaler_enabled):
            logits = model(x)
            loss   = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss    += loss.item() * x.size(0)
        mask           = (y != 0)
        total_correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
        total_tokens  += mask.sum().item()
    return total_loss / len(val_loader.dataset), total_correct / max(total_tokens, 1) * 100

# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(path, model, optimizer, scaler, tokenizer, epoch, global_step,
                    best_val_loss, early_stopping, history, config):
    torch.save({
        'version':         SCRIPT_VERSION,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state':    scaler.state_dict(),
        'tokenizer_char2idx': tokenizer.char2idx,
        'epoch':           epoch,
        'global_step':     global_step,
        'best_val_loss':   best_val_loss,
        'early_stopping':  early_stopping.state_dict(),
        'history':         history,
        'config': {
            'vocab_size':  tokenizer.vocab_size,
            'embed_dim':   config.EMBED_DIM,
            'num_heads':   config.NUM_HEADS,
            'num_layers':  config.NUM_LAYERS,
            'ff_dim':      config.FF_DIM,
            'max_seq_len': config.MAX_SEQ_LEN,
            'dropout':     config.DROPOUT,
        }
    }, path)


def load_checkpoint(path, model, optimizer, scaler, early_stopping, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get('version', 0) != SCRIPT_VERSION:
        print(f"   Checkpoint v{ckpt.get('version')} incompatible avec v{SCRIPT_VERSION}")
        return None
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scaler.load_state_dict(ckpt['scaler_state'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    early_stopping.load_state_dict(ckpt['early_stopping'])
    return ckpt['epoch'], ckpt['global_step'], ckpt['best_val_loss'], ckpt['history']

# ============================================================
# LR SCHEDULE
# ============================================================

def get_lr(step, warmup_steps, max_lr, cycle_length=5000, cycle_decay=0.80):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    effective  = step - warmup_steps
    cycle_num  = effective // cycle_length
    in_cycle   = effective % cycle_length
    cycle_lr   = max_lr * (cycle_decay ** cycle_num)
    min_lr     = cycle_lr * 0.01
    return min_lr + 0.5 * (cycle_lr - min_lr) * (1 + math.cos(math.pi * in_cycle / cycle_length))

# ============================================================
# TRAINING
# ============================================================

_interrupted = False

def train_model(model, train_data, val_loader, eval_passwords, tokenizer, config,
                start_epoch=0, global_step=0, best_val_loss=float('inf'),
                history=None, optimizer=None, early_stopping=None, scaler=None):
    """train_data : tensor GPU (mode rapide) ou DataLoader (fallback)."""
    global _interrupted
    device  = config.DEVICE
    model   = model.to(device)
    use_amp = config.USE_AMP and device == 'cuda'

    gpu_mode = isinstance(train_data, torch.Tensor)
    if gpu_mode:
        n_train         = train_data.size(0)
        num_batches     = n_train // config.BATCH_SIZE
    else:
        num_batches     = len(train_data)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.95)
        )
    if scaler is None:
        scaler = GradScaler(enabled=use_amp)

    pad_idx   = tokenizer.char2idx[CharTokenizer.PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=config.LABEL_SMOOTHING)

    if early_stopping is None:
        early_stopping = EarlyStopping(patience=config.PATIENCE, min_delta=config.MIN_DELTA)

    accum            = config.GRAD_ACCUM_STEPS
    steps_per_epoch  = num_batches // accum
    if history is None:
        history = []

    def signal_handler(sig, frame):
        global _interrupted
        _interrupted = True
        print("\n\n   Ctrl+C detecte! Sauvegarde checkpoint...")

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    print(f"\n{'='*60}")
    print(f"ENTRAINEMENT v{SCRIPT_VERSION} {'(REPRISE)' if start_epoch > 0 else ''}")
    print(f"{'='*60}")
    print(f"   Device: {device.upper()} | AMP float16: {'ON' if use_amp else 'OFF'}")
    print(f"   Mode: {'GPU-RESIDENT (zero copie CPU->GPU)' if gpu_mode else 'DataLoader'}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory//1024**3}GB")
    print(f"   Parametres: {model.count_parameters():,} ({model.count_parameters()*4/1024/1024:.1f} MB)")
    print(f"   Epochs: {start_epoch+1} -> {config.EPOCHS} max")
    print(f"   Batch effective: {config.BATCH_SIZE} x {accum} = {config.BATCH_SIZE * accum}")
    print(f"   Steps/epoch: {steps_per_epoch:,}")
    print(f"   Early stopping: {early_stopping.status()}")
    print(f"   Train: {num_batches:,} batches | Val: {len(val_loader):,} batches")
    print(f"{'='*60}")

    for epoch in range(start_epoch, config.EPOCHS):
        if _interrupted:
            break

        model.train()
        train_loss    = 0.0
        train_correct = 0
        train_total   = 0
        epoch_start   = time.time()
        optimizer.zero_grad(set_to_none=True)

        batch_iter = gpu_batch_iter(train_data, config.BATCH_SIZE) if gpu_mode else train_data
        for batch_idx, (x, y) in enumerate(batch_iter):
            if _interrupted:
                break

            if not gpu_mode:
                x = x.to(device, dtype=torch.long, non_blocking=True)
                y = y.to(device, dtype=torch.long, non_blocking=True)

            opt_step = global_step // accum
            lr = get_lr(opt_step, config.WARMUP_STEPS, config.LEARNING_RATE,
                        config.LR_CYCLE_LENGTH, config.LR_CYCLE_DECAY)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss   = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
                loss   = loss / accum

            scaler.scale(loss).backward()

            with torch.no_grad():
                mask           = (y != pad_idx)
                train_correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
                train_total   += mask.sum().item()

            if (batch_idx + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accum
            global_step += 1

            if batch_idx % 200 == 0:
                acc     = train_correct / max(train_total, 1) * 100
                elapsed = time.time() - epoch_start
                ms_per_batch = elapsed / max(batch_idx, 1) * 1000
                print(f"   Epoch {epoch+1}/{config.EPOCHS} | Batch {batch_idx}/{num_batches} | "
                      f"Loss: {loss.item()*accum:.4f} | Acc: {acc:.1f}% | "
                      f"LR: {lr:.2e} | {ms_per_batch:.0f}ms/batch")

        if _interrupted:
            print(f"\n   Sauvegarde checkpoint epoch {epoch+1}...")
            save_checkpoint(config.CHECKPOINT_PATH, model, optimizer, scaler,
                            tokenizer, epoch, global_step, best_val_loss,
                            early_stopping, history, config)
            print(f"   Checkpoint sauvegarde!")
            break

        avg_train_loss = train_loss / num_batches
        train_acc      = train_correct / max(train_total, 1) * 100
        val_loss, val_acc = evaluate_val(model, val_loader, criterion, device, use_amp)
        elapsed        = time.time() - epoch_start
        overfit_gap    = val_loss - avg_train_loss

        print(f"\n   +{'='*50}+")
        print(f"   | Epoch {epoch+1:2d}/{config.EPOCHS} | Temps: {elapsed:.0f}s")
        print(f"   | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"   | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"   | Gap: {overfit_gap:.4f} {'(OVERFIT!)' if overfit_gap > 0.3 else '(OK)'}")
        print(f"   | Early Stop: {early_stopping.status()}")
        print(f"   +{'='*50}+")

        # Quick coverage check toutes les 5 epochs
        quick_coverage = 0.0
        if (epoch + 1) % 5 == 0 and eval_passwords:
            print(f"   Quick eval (50k candidats)...")
            qg   = generate_passwords(model, tokenizer, 50_000, 0.8, 50, 0.92, device, 4096)
            es   = set(eval_passwords[:10_000])  # sample de 10k pour rapidite
            qm   = qg & es
            quick_coverage = len(qm) / len(es) * 100
            print(f"   Coverage (sample 10k eval): {len(qm)}/{len(es)} = {quick_coverage:.2f}%")
            if qm:
                print(f"   Exemples: {list(qm)[:8]}")

        history.append({
            'epoch': epoch + 1, 'train_loss': round(avg_train_loss, 4),
            'train_acc': round(train_acc, 2), 'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 2), 'overfit_gap': round(overfit_gap, 4),
            'lr': round(lr, 8), 'time_sec': round(elapsed, 1),
            'quick_coverage': round(quick_coverage, 2)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'version':    SCRIPT_VERSION,
                'model_state': model.state_dict(),
                'tokenizer_char2idx': tokenizer.char2idx,
                'epoch': epoch + 1, 'val_loss': val_loss,
                'config': {
                    'vocab_size': tokenizer.vocab_size, 'embed_dim': config.EMBED_DIM,
                    'num_heads': config.NUM_HEADS, 'num_layers': config.NUM_LAYERS,
                    'ff_dim': config.FF_DIM, 'max_seq_len': config.MAX_SEQ_LEN,
                    'dropout': config.DROPOUT
                }
            }, config.MODEL_PATH)
            print(f"   Meilleur modele sauvegarde (val_loss={val_loss:.4f})")

        save_checkpoint(config.CHECKPOINT_PATH, model, optimizer, scaler,
                        tokenizer, epoch + 1, global_step, best_val_loss,
                        early_stopping, history, config)
        print(f"   Checkpoint epoch {epoch+1} sauvegarde (RAM disk)")

        # Synchro vers NFS toutes les 3 epochs
        if (epoch + 1) % 3 == 0:
            import shutil
            nfs_models = os.path.join(config.BASE_DIR, 'output', 'models')
            os.makedirs(nfs_models, exist_ok=True)
            shutil.copy2(config.CHECKPOINT_PATH, os.path.join(nfs_models, 'v6_checkpoint.pt'))
            if os.path.exists(config.MODEL_PATH):
                shutil.copy2(config.MODEL_PATH, os.path.join(nfs_models, 'v6_model.pt'))
            print(f"   Synchro NFS effectuee (epoch {epoch+1})")

        if early_stopping(val_loss):
            print(f"\n   EARLY STOPPING a epoch {epoch+1}!")
            break

    signal.signal(signal.SIGINT, old_handler)
    return history, optimizer, early_stopping, scaler

# ============================================================
# EVALUATION COMPLETE
# ============================================================

def full_evaluation(model, tokenizer, eval_passwords, config):
    device    = config.DEVICE
    eval_set  = set(eval_passwords)
    results   = {}
    all_generated = set()

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"Eval set: {len(eval_set):,} passwords")
    print(f"{'='*60}")

    for temp in config.TEMPERATURE:
        print(f"\nTemp={temp} | top_k={config.TOP_K} | top_p={config.TOP_P}")
        print(f"   Generation de {config.NUM_GENERATE:,} candidats...")

        t0        = time.time()
        generated = generate_passwords(model, tokenizer, config.NUM_GENERATE,
                                       temp, config.TOP_K, config.TOP_P,
                                       device, config.GEN_BATCH, config.GEN_STALE_LIMIT)
        gen_time  = time.time() - t0
        all_generated |= generated

        matches  = generated & eval_set
        coverage = len(matches) / len(eval_set) * 100
        lengths  = [len(p) for p in generated]

        result = {
            'temperature':      temp,
            'total_generated':  len(generated),
            'gen_time_sec':     round(gen_time, 1),
            'coverage': {
                'matches': len(matches), 'total_eval': len(eval_set),
                'pct': round(coverage, 4)
            },
            'avg_length':        round(sum(lengths) / max(len(lengths), 1), 2),
            'matched_examples':  sorted(list(matches))[:50],
            'sample_generated':  random.sample(list(generated), min(20, len(generated)))
        }
        results[f'temp_{temp}'] = result

        print(f"   Generes: {len(generated):,} en {gen_time:.1f}s")
        print(f"   Coverage: {len(matches)}/{len(eval_set)} = {coverage:.2f}%")
        if matches:
            print(f"   Exemples: {sorted(list(matches))[:10]}")

    all_matches = all_generated & eval_set
    combined    = len(all_matches) / len(eval_set) * 100

    print(f"\n{'='*60}")
    print(f"COVERAGE COMBINE: {len(all_matches)}/{len(eval_set)} = {combined:.2f}%")
    print(f"Total unique genere: {len(all_generated):,}")
    print(f"{'='*60}")

    with open(config.GEN_PATH, 'w', encoding='utf-8') as f:
        f.write(f"# MATCHES ({len(all_matches)}/{len(eval_set)} = {combined:.2f}% coverage)\n")
        for pwd in sorted(all_matches):
            f.write(pwd + '\n')
        f.write(f"\n# TOUS LES GENERES ({len(all_generated):,} uniques)\n")
        for pwd in sorted(all_generated):
            f.write(pwd + '\n')

    results['combined'] = {
        'total_unique_generated': len(all_generated),
        'coverage_matches': len(all_matches),
        'coverage_pct': round(combined, 4),
        'all_matched': sorted(list(all_matches))
    }
    return results

# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    config        = Config()
    reset         = '--reset' in sys.argv
    generate_only = '--generate-only' in sys.argv

    # Creer le dossier fast (RAM disk) et copier checkpoint existant si besoin
    NFS_MODEL = os.path.join(config.BASE_DIR, 'output', 'models', 'v6_model.pt')
    NFS_CKPT  = os.path.join(config.BASE_DIR, 'output', 'models', 'v6_checkpoint.pt')
    os.makedirs(config._FAST_DIR, exist_ok=True)
    if not os.path.exists(config.CHECKPOINT_PATH) and os.path.exists(NFS_CKPT):
        import shutil
        print(f"Copie checkpoint NFS -> RAM disk ({NFS_CKPT} -> {config.CHECKPOINT_PATH})")
        shutil.copy2(NFS_CKPT, config.CHECKPOINT_PATH)
    if not os.path.exists(config.MODEL_PATH) and os.path.exists(NFS_MODEL):
        import shutil
        print(f"Copie model NFS -> RAM disk")
        shutil.copy2(NFS_MODEL, config.MODEL_PATH)

    print("=" * 60)
    print(f"PasswordTransformer v{SCRIPT_VERSION} — RockYou 14M")
    print(f"Checkpoints: {config._FAST_DIR}")
    print(f"Device: {config.DEVICE.upper()} | AMP: {config.USE_AMP}")
    print(f"Archi: {config.NUM_LAYERS}L-{config.NUM_HEADS}H-{config.EMBED_DIM}D | SwiGLU+RMSNorm+KV-cache")
    print("=" * 60)

    def load_passwords(path, max_len=None):
        ml = (max_len or Config.MAX_SEQ_LEN) - 2
        pwds = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                pwd = line.strip()
                if pwd and 2 <= len(pwd) <= ml:
                    pwds.append(pwd)
        return pwds

    print("\nChargement des donnees...")
    all_train_pwds = load_passwords(config.TRAIN_PATH)
    eval_pwds      = load_passwords(config.EVAL_PATH)
    print(f"   Train brut: {len(all_train_pwds):,} | Eval: {len(eval_pwds):,}")

    random.shuffle(all_train_pwds)
    val_size   = int(len(all_train_pwds) * config.VAL_SPLIT)
    val_pwds   = all_train_pwds[:val_size]
    train_pwds = all_train_pwds[val_size:]
    print(f"   Split -> Train: {len(train_pwds):,} | Val: {len(val_pwds):,}")

    has_checkpoint = os.path.exists(config.CHECKPOINT_PATH) and not reset

    if reset:
        for p in [config.CHECKPOINT_PATH, config.MODEL_PATH]:
            if os.path.exists(p):
                os.remove(p)
                print(f"   Supprime: {os.path.basename(p)}")
        has_checkpoint = False

    if has_checkpoint:
        print(f"\nCheckpoint trouve, verification...")
        preview = torch.load(config.CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        if preview.get('version', 0) != SCRIPT_VERSION:
            print(f"   Incompatible (v{preview.get('version')} != v{SCRIPT_VERSION}), restart")
            os.remove(config.CHECKPOINT_PATH)
            if os.path.exists(config.MODEL_PATH):
                os.remove(config.MODEL_PATH)
            has_checkpoint = False
        del preview

    if has_checkpoint:
        print("   Reprise de l'entrainement...")
        ckpt      = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)
        tokenizer = CharTokenizer().from_dict(ckpt['tokenizer_char2idx'])
        model     = PasswordTransformer(
            tokenizer.vocab_size, config.EMBED_DIM, config.NUM_HEADS,
            config.NUM_LAYERS, config.FF_DIM, config.MAX_SEQ_LEN, config.DROPOUT
        ).to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                       weight_decay=config.WEIGHT_DECAY, betas=(0.9, 0.95))
        scaler        = GradScaler(enabled=config.USE_AMP and config.DEVICE == 'cuda')
        early_stopping = EarlyStopping(config.PATIENCE, config.MIN_DELTA)
        result        = load_checkpoint(config.CHECKPOINT_PATH, model, optimizer,
                                        scaler, early_stopping, config.DEVICE)
        if result is None:
            has_checkpoint = False
        else:
            start_epoch, global_step, best_val_loss, history = result
            print(f"   Reprise epoch {start_epoch+1}, step {global_step}")
            print(f"   Best val loss: {best_val_loss:.4f} | {early_stopping.status()}")

    if not has_checkpoint:
        print("\nConstruction du tokenizer...")
        tokenizer = CharTokenizer()
        tokenizer.fit(all_train_pwds + eval_pwds)
        model = PasswordTransformer(
            tokenizer.vocab_size, config.EMBED_DIM, config.NUM_HEADS,
            config.NUM_LAYERS, config.FF_DIM, config.MAX_SEQ_LEN, config.DROPOUT
        )
        print(f"   Parametres: {model.count_parameters():,} ({model.count_parameters()*4/1024/1024:.1f} MB)")
        optimizer      = None
        scaler         = None
        early_stopping = None
        start_epoch    = 0
        global_step    = 0
        best_val_loss  = float('inf')
        history        = []

    print("\nPre-tensorisation...")
    train_tensor = tokenizer.encode_all(train_pwds)
    val_tensor   = tokenizer.encode_all(val_pwds)
    print(f"   Train: {train_tensor.shape} | Val: {val_tensor.shape}")

    # Charge tout le dataset train en VRAM une seule fois (zero copie CPU->GPU par batch)
    print(f"\nChargement tensor train -> {config.DEVICE.upper()}...")
    t0 = time.time()
    train_gpu = train_tensor.to(config.DEVICE, dtype=torch.long)
    vram_mb   = train_gpu.element_size() * train_gpu.nelement() / 1024 / 1024
    print(f"   {vram_mb:.0f} MB charges en {time.time()-t0:.1f}s — zero copie par batch desormais")
    del train_tensor  # libere la RAM CPU

    val_dataset  = PasswordDataset(val_tensor)
    val_loader   = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    if not generate_only:
        history, optimizer, early_stopping, scaler = train_model(
            model, train_gpu, val_loader, eval_pwds, tokenizer, config,
            start_epoch, global_step, best_val_loss, history, optimizer,
            early_stopping, scaler
        )

    # Charger le meilleur modele
    if os.path.exists(config.MODEL_PATH):
        print("\nChargement du meilleur modele...")
        best = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(best['model_state'])
        model = model.to(config.DEVICE)
        print(f"   Epoch {best['epoch']}, val_loss={best['val_loss']:.4f}")

    eval_results = full_evaluation(model, tokenizer, eval_pwds, config)

    if not ('--quick' in sys.argv):
        final = {
            'version': SCRIPT_VERSION,
            'config': {
                'architecture': 'SwiGLU+RMSNorm+SDPA+KV-cache+AMP',
                'device': config.DEVICE, 'embed_dim': config.EMBED_DIM,
                'num_heads': config.NUM_HEADS, 'num_layers': config.NUM_LAYERS,
                'ff_dim': config.FF_DIM, 'batch_size': config.BATCH_SIZE,
                'grad_accum': config.GRAD_ACCUM_STEPS, 'epochs_run': len(history),
                'params': model.count_parameters()
            },
            'training_history': history,
            'evaluation': eval_results
        }
        with open(config.OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final, f, indent=2, ensure_ascii=False, default=str)

    # Synchro finale RAM disk -> NFS
    import shutil
    nfs_models = os.path.join(config.BASE_DIR, 'output', 'models')
    os.makedirs(nfs_models, exist_ok=True)
    for fname in ['v6_checkpoint.pt', 'v6_model.pt']:
        src = os.path.join(config._FAST_DIR, fname)
        dst = os.path.join(nfs_models, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"   Synchro finale: {fname} -> NFS")

    combined = eval_results.get('combined', {})
    print(f"\n{'='*60}")
    print(f"RESUME FINAL v{SCRIPT_VERSION}")
    print(f"   Epochs: {len(history)}")
    print(f"   Coverage: {combined.get('coverage_pct', 0):.2f}%")
    print(f"   Matches: {combined.get('coverage_matches', 0)}/{len(eval_pwds):,}")
    print(f"   Total genere: {combined.get('total_unique_generated', 0):,}")
    print(f"   Resultats: {config.OUTPUT_PATH}")
    print(f"   Modele: {nfs_models}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
