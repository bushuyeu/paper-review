#!/usr/bin/env python3
"""
============================================================================
STEP 3b: Figure 3 Experiment with OFFICIAL + CUSTOM Tokenizers
============================================================================

Uses the SAME tokenizers as step3:
    - FAST+: physical-intelligence/fast (HuggingFace)
    - Custom FAST: trained via .fit() on berkeley_cable_routing
    - Naive: OpenVLA ActionTokenizer

This ensures consistency across all experiments.

Key insight being tested (from paper Figure 3):
    At higher sampling rates, naive tokenization produces tokens that are
    highly correlated (each token ≈ previous token). This makes next-token
    prediction trivial but actual sequence prediction poor.
    
    DCT tokenization decorrelates the signal, providing better learning.

============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Import official tokenizers and data loading from step3
from step3_fast_tokenization import (
    load_trajectories,
    load_official_fast_tokenizer,
    load_openvla_action_tokenizer,
    load_or_train_custom_fast_tokenizer,
    DEFAULT_CUSTOM_FAST_SAVE_DIR,
    DEFAULT_CHUNK_STEPS,
)

OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Custom FAST settings (match step3 defaults)
CUSTOM_FAST_VOCAB_SIZE = 1024
CUSTOM_FAST_SCALE = 10.0
CUSTOM_FAST_CHUNK_STEPS = DEFAULT_CHUNK_STEPS
CUSTOM_FAST_STRIDE = DEFAULT_CHUNK_STEPS
CUSTOM_FAST_SAVE_DIR = DEFAULT_CUSTOM_FAST_SAVE_DIR
CUSTOM_FAST_FORCE_RETRAIN = False


# ============================================================================
# NORMALIZATION + CUSTOM FAST TRAINING HELPERS
# ============================================================================

def compute_action_normalization(trajectories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute robust normalization bounds from all actions."""
    all_actions = np.vstack(trajectories)
    q1 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)
    return q1, q99


def normalize_actions(actions: np.ndarray, q1: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Normalize actions to [-1, 1] using robust percentiles."""
    range_val = np.maximum(q99 - q1, 1e-8)
    return np.clip(2 * (actions - q1) / range_val - 1, -1, 1)


def build_custom_fast_training_chunks(
    trajectories: List[np.ndarray],
    q1: np.ndarray,
    q99: np.ndarray,
    *,
    chunk_steps: int,
    stride: int,
) -> np.ndarray:
    """Create normalized fixed-length chunks for custom FAST training."""
    chunks: list[np.ndarray] = []
    for traj in trajectories:
        if len(traj) < chunk_steps:
            continue
        for start in range(0, len(traj) - chunk_steps + 1, stride):
            chunk = traj[start:start + chunk_steps]
            chunks.append(normalize_actions(chunk, q1, q99))
    if not chunks:
        raise RuntimeError("No chunks available for custom FAST training.")
    return np.stack(chunks)


def infer_fast_vocab_size(processor, fallback: int) -> int:
    """Best-effort vocab size lookup for FAST processors."""
    bpe = getattr(processor, "bpe_tokenizer", None)
    if bpe is not None and hasattr(bpe, "vocab_size"):
        try:
            return int(bpe.vocab_size)
        except Exception:
            pass
    if hasattr(processor, "vocab_size"):
        try:
            return int(processor.vocab_size)
        except Exception:
            pass
    return fallback


# ============================================================================
# TOKENIZER WRAPPERS (for encode/decode with official tokenizers)
# ============================================================================

class OfficialFASTWrapper:
    """
    Wrapper around official FAST+ tokenizer that provides encode/decode.
    
    Note: FAST+ doesn't provide a built-in decode, so we measure prediction
    quality in TOKEN space (cross-entropy) rather than continuous space.
    """
    
    def __init__(self):
        self.tokenizer = load_official_fast_tokenizer()
        self._vocab_size = infer_fast_vocab_size(self.tokenizer, 1024)
    
    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode normalized chunk (T, D) to tokens."""
        # FAST+ expects (batch, T, D)
        result = self.tokenizer(chunk[np.newaxis, ...])
        tokens = np.array(result[0] if isinstance(result, list) else result)
        return tokens.astype(np.int64)
    
    @property
    def vocab_size(self):
        return self._vocab_size


class CustomFASTWrapper:
    """
    Wrapper around custom FAST tokenizer trained via .fit().

    Expects normalized input in [-1, 1] (same as OfficialFASTWrapper).
    """

    def __init__(self, tokenizer, vocab_size: Optional[int] = None):
        self.tokenizer = tokenizer
        self._vocab_size = (
            infer_fast_vocab_size(tokenizer, CUSTOM_FAST_VOCAB_SIZE)
            if vocab_size is None
            else vocab_size
        )

    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode normalized chunk (T, D) to tokens."""
        result = self.tokenizer(chunk[np.newaxis, ...])
        tokens = np.array(result[0] if isinstance(result, list) else result)
        return tokens.astype(np.int64)

    @property
    def vocab_size(self):
        return self._vocab_size


class OfficialNaiveWrapper:
    """
    Wrapper around official OpenVLA ActionTokenizer.
    
    Provides consistent interface with FAST+ wrapper.
    """
    
    def __init__(self):
        openvla_tok = load_openvla_action_tokenizer()
        self.tokenizer = openvla_tok
        self.min_action = float(openvla_tok.min_action)
        self.max_action = float(openvla_tok.max_action)
        self.bins = openvla_tok.bins  # This is a numpy array of bin edges
        self.n_bins = 256
        self._vocab_size = 256
    
    def encode(self, chunk: np.ndarray) -> np.ndarray:
        """Encode chunk (T, D) to tokens."""
        # Clip to valid range
        chunk_clipped = np.clip(chunk, self.min_action, self.max_action)
        # Digitize each value
        tokens = np.digitize(chunk_clipped, self.bins)
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        return tokens.astype(np.int64).flatten()
    
    @property
    def vocab_size(self):
        return self._vocab_size


# ============================================================================
# DATASET
# ============================================================================

class CableDataset(Dataset):
    """
    Dataset for prediction task on real cable trajectories.
    
    Resamples trajectories to different lengths to simulate different
    control frequencies (matching Figure 3's experimental setup).
    """
    
    def __init__(
        self,
        trajectories: List[np.ndarray],
        tokenizer,
        target_length: int,
        base_length: int = 40,
        q1: Optional[np.ndarray] = None,
        q99: Optional[np.ndarray] = None,
    ):
        self.tokenizer = tokenizer
        self.target_length = target_length
        
        # Compute normalization params
        if q1 is None or q99 is None:
            self.q1, self.q99 = compute_action_normalization(trajectories)
        else:
            self.q1, self.q99 = q1, q99
        
        # Process trajectories
        self.samples = []
        for traj in trajectories:
            if len(traj) < base_length:
                continue
            
            # Take segment and normalize
            segment = traj[:base_length]
            segment_norm = self._normalize(segment)
            
            # Resample to target length
            resampled = self._resample(segment_norm, target_length)
            
            # Tokenize
            try:
                tokens = tokenizer.encode(resampled)
                if len(tokens) > 0:
                    self.samples.append({
                        'trajectory': resampled,
                        'tokens': tokens,
                    })
            except Exception as e:
                continue  # Skip problematic samples
        
        print(f"Dataset: {len(self.samples)} samples, {target_length} steps each")
    
    def _normalize(self, traj: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] range."""
        return normalize_actions(traj, self.q1, self.q99)
    
    def _resample(self, traj: np.ndarray, target_len: int) -> np.ndarray:
        """Resample trajectory via interpolation."""
        T, D = traj.shape
        t_orig = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, target_len)
        
        resampled = np.zeros((target_len, D))
        for d in range(D):
            f = interp1d(t_orig, traj[:, d], kind='linear')
            resampled[:, d] = f(t_new)
        
        return resampled
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'tokens': torch.tensor(sample['tokens'], dtype=torch.long),
            'trajectory': torch.tensor(sample['trajectory'], dtype=torch.float32),
        }


# ============================================================================
# MODEL
# ============================================================================

class SmallTransformer(nn.Module):
    """Small autoregressive transformer for next-token prediction."""
    
    def __init__(self, vocab_size: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, max_len: int = 2048):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        
        x = self.embed(tokens) + self.pos_embed(pos)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        
        return self.head(x)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def pad_token_batch(batch, pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad variable-length token sequences to a common length."""
    tokens_list = [item["tokens"] for item in batch]
    lengths = [t.shape[0] for t in tokens_list]
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        return {"tokens": torch.empty((0, 0), dtype=torch.long)}

    padded = torch.full((len(tokens_list), max_len), pad_id, dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        padded[i, :tokens.shape[0]] = tokens
    return {"tokens": padded}


def train_and_evaluate(
    trajectories: List[np.ndarray],
    tokenizer,
    target_length: int,
    *,
    q1: Optional[np.ndarray] = None,
    q99: Optional[np.ndarray] = None,
    n_epochs: int = 30,
    batch_size: int = 32,
) -> Dict:
    """
    Train autoregressive model and evaluate prediction quality.
    
    Since official tokenizers don't provide easy decode, we measure:
    1. Final cross-entropy loss (lower = better token prediction)
    2. Token prediction accuracy (higher = better)
    
    The key insight from Figure 3 is that naive tokenization leads to
    WORSE prediction as frequency increases, while DCT stays stable.
    """
    
    # Create dataset
    dataset = CableDataset(
        trajectories,
        tokenizer,
        target_length,
        q1=q1,
        q99=q99,
    )
    
    if len(dataset) < 20:
        print(f"Warning: Only {len(dataset)} samples")
        return {'loss': float('nan'), 'accuracy': float('nan')}
    
    # Split
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    pad_id = int(tokenizer.vocab_size)
    collate_fn = lambda batch: pad_token_batch(batch, pad_id)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)
    
    # Model
    model = SmallTransformer(
        vocab_size=tokenizer.vocab_size + 1,
        d_model=64, n_heads=4, n_layers=3,
        max_len=2048
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # Training
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            tokens = batch['tokens'].to(DEVICE)
            
            # Skip if sequence too short
            if tokens.shape[1] < 2:
                continue
            
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            logits = model(input_tokens)
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size + 1),
                target_tokens.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0 and n_batches > 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")
    
    # Evaluation
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(DEVICE)
            
            if tokens.shape[1] < 2:
                continue
            
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            logits = model(input_tokens)
            
            # Loss
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size + 1),
                target_tokens.reshape(-1)
            )
            valid = target_tokens != pad_id
            total_loss += loss.item() * valid.sum().item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == target_tokens) & valid).sum().item()
            total_tokens += valid.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('nan')
    accuracy = total_correct / total_tokens if total_tokens > 0 else float('nan')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(trajectories: List[np.ndarray]):
    """
    Run Figure 3 style experiment with official + custom tokenizers.
    """
    
    print()
    print("=" * 70)
    print("FIGURE 3 EXPERIMENT - OFFICIAL + CUSTOM TOKENIZERS")
    print("=" * 70)
    print()
    print("Using tokenizers (same as step3):")
    print("  - FAST+: physical-intelligence/fast")
    print("  - Custom FAST: trained via .fit() on Berkeley cable data")
    print("  - Naive: OpenVLA ActionTokenizer")
    print()
    print("Metric: Cross-entropy loss on next-token prediction")
    print("        (since official tokenizers don't provide decode)")
    print()
    print("Expected result (from paper):")
    print("  - Naive: loss INCREASES with sampling rate (tokens too correlated)")
    print("  - FAST+: loss stays STABLE (tokens decorrelated by DCT)")
    print()

    # Shared normalization (used for dataset + custom FAST training)
    q1, q99 = compute_action_normalization(trajectories)

    # Initialize tokenizers once
    print("Loading tokenizers...")
    naive_tok = OfficialNaiveWrapper()
    fast_tok = OfficialFASTWrapper()

    custom_tok: Optional[CustomFASTWrapper] = None
    try:
        print("Preparing custom FAST tokenizer...")
        custom_chunks = build_custom_fast_training_chunks(
            trajectories,
            q1,
            q99,
            chunk_steps=CUSTOM_FAST_CHUNK_STEPS,
            stride=CUSTOM_FAST_STRIDE,
        )
        custom_processor = load_or_train_custom_fast_tokenizer(
            custom_chunks,
            vocab_size=CUSTOM_FAST_VOCAB_SIZE,
            scale=CUSTOM_FAST_SCALE,
            save_dir=CUSTOM_FAST_SAVE_DIR,
            force_retrain=CUSTOM_FAST_FORCE_RETRAIN,
        )
        custom_tok = CustomFASTWrapper(custom_processor)
        print("Custom FAST tokenizer ready")
    except Exception as exc:
        print(f"Custom FAST unavailable: {exc}")
    
    # Different sampling rates (trajectory lengths)
    sample_lengths = [10, 20, 40, 80]
    
    results = {
        'sample_lengths': sample_lengths,
        'naive': {'loss': [], 'accuracy': []},
        'fast': {'loss': [], 'accuracy': []},
        'custom': {'loss': [], 'accuracy': []},
    }
    
    for length in sample_lengths:
        print()
        print(f"{'='*60}")
        print(f"SAMPLE LENGTH: {length} steps")
        print(f"{'='*60}")
        
        # Naive tokenization
        print("\n--- Naive (OpenVLA) ---")
        try:
            naive_result = train_and_evaluate(
                trajectories,
                naive_tok,
                target_length=length,
                q1=q1,
                q99=q99,
                n_epochs=30,
            )
            results['naive']['loss'].append(naive_result['loss'])
            results['naive']['accuracy'].append(naive_result['accuracy'])
            print(f"Loss: {naive_result['loss']:.4f}, Accuracy: {naive_result['accuracy']:.2%}")
        except Exception as e:
            print(f"Error: {e}")
            results['naive']['loss'].append(float('nan'))
            results['naive']['accuracy'].append(float('nan'))
        
        # FAST+ tokenization
        print("\n--- FAST+ (Official) ---")
        try:
            fast_result = train_and_evaluate(
                trajectories,
                fast_tok,
                target_length=length,
                q1=q1,
                q99=q99,
                n_epochs=30,
            )
            results['fast']['loss'].append(fast_result['loss'])
            results['fast']['accuracy'].append(fast_result['accuracy'])
            print(f"Loss: {fast_result['loss']:.4f}, Accuracy: {fast_result['accuracy']:.2%}")
        except Exception as e:
            print(f"Error: {e}")
            results['fast']['loss'].append(float('nan'))
            results['fast']['accuracy'].append(float('nan'))

        # Custom FAST tokenization
        print("\n--- Custom FAST (Trained) ---")
        if custom_tok is None:
            print("Custom FAST unavailable; skipping.")
            results['custom']['loss'].append(float('nan'))
            results['custom']['accuracy'].append(float('nan'))
        else:
            try:
                custom_result = train_and_evaluate(
                    trajectories,
                    custom_tok,
                    target_length=length,
                    q1=q1,
                    q99=q99,
                    n_epochs=30,
                )
                results['custom']['loss'].append(custom_result['loss'])
                results['custom']['accuracy'].append(custom_result['accuracy'])
                print(f"Loss: {custom_result['loss']:.4f}, Accuracy: {custom_result['accuracy']:.2%}")
            except Exception as e:
                print(f"Error: {e}")
                results['custom']['loss'].append(float('nan'))
                results['custom']['accuracy'].append(float('nan'))
    
    return results


def print_results(results: Dict):
    """Print formatted results."""
    
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Length':<8} {'Naive Loss':<12} {'FAST+ Loss':<12} {'Custom Loss':<12} "
          f"{'Naive Acc':<10} {'FAST+ Acc':<10} {'Custom Acc':<10}")
    print("-" * 70)
    
    for i, length in enumerate(results['sample_lengths']):
        naive_loss = results['naive']['loss'][i]
        fast_loss = results['fast']['loss'][i]
        custom_loss = results['custom']['loss'][i]
        naive_acc = results['naive']['accuracy'][i]
        fast_acc = results['fast']['accuracy'][i]
        custom_acc = results['custom']['accuracy'][i]
        
        print(f"{length:<8} {naive_loss:<12.4f} {fast_loss:<12.4f} {custom_loss:<12.4f} "
              f"{naive_acc:<10.2%} {fast_acc:<10.2%} {custom_acc:<10.2%}")
    
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    naive_losses = [x for x in results['naive']['loss'] if not np.isnan(x)]
    fast_losses = [x for x in results['fast']['loss'] if not np.isnan(x)]
    custom_losses = [x for x in results['custom']['loss'] if not np.isnan(x)]
    
    if len(naive_losses) >= 2 and len(fast_losses) >= 2:
        naive_trend = naive_losses[-1] - naive_losses[0]
        fast_trend = fast_losses[-1] - fast_losses[0]
        custom_trend = custom_losses[-1] - custom_losses[0] if len(custom_losses) >= 2 else float('nan')
        
        print()
        print(f"  Naive loss change (short→long): {naive_trend:+.4f}")
        print(f"  FAST+ loss change (short→long): {fast_trend:+.4f}")
        if len(custom_losses) >= 2:
            print(f"  Custom loss change (short→long): {custom_trend:+.4f}")
        
        if naive_trend > fast_trend:
            print()
            print("  ✅ VALIDATES PAPER'S FIGURE 3:")
            print("     Naive prediction degrades more at higher sampling rates")
            print("     FAST+ remains more stable")
        else:
            print()
            print("  ⚠️  Results don't clearly match paper's Figure 3")
            print("     (May need more epochs or different hyperparameters)")


def create_plot(results: Dict):
    """Create Figure 3 style plot."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    lengths = results['sample_lengths']
    
    # Plot 1: Loss
    ax1 = axes[0]
    ax1.plot(lengths, results['naive']['loss'], 'o-', color='orange',
             label='Naive (OpenVLA)', linewidth=2, markersize=8)
    ax1.plot(lengths, results['fast']['loss'], 's-', color='green',
             label='FAST+ (Official)', linewidth=2, markersize=8)
    ax1.plot(lengths, results['custom']['loss'], 'd-', color='blue',
             label='Custom FAST (Trained)', linewidth=2, markersize=8)
    ax1.set_xlabel('Sample Length (steps)')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Prediction Loss vs Sampling Rate\n(Lower = Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2 = axes[1]
    ax2.plot(lengths, results['naive']['accuracy'], 'o-', color='orange',
             label='Naive (OpenVLA)', linewidth=2, markersize=8)
    ax2.plot(lengths, results['fast']['accuracy'], 's-', color='green',
             label='FAST+ (Official)', linewidth=2, markersize=8)
    ax2.plot(lengths, results['custom']['accuracy'], 'd-', color='blue',
             label='Custom FAST (Trained)', linewidth=2, markersize=8)
    ax2.set_xlabel('Sample Length (steps)')
    ax2.set_ylabel('Token Prediction Accuracy')
    ax2.set_title('Prediction Accuracy vs Sampling Rate\n(Higher = Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure3_official_tokenizers.png', dpi=150)
    print(f"\nFigure saved to: {OUTPUT_DIR / 'figure3_official_tokenizers.png'}")
    plt.close()


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " FIGURE 3 EXPERIMENT - OFFICIAL + CUSTOM ".center(68) + "║")
    print("║" + " FAST+ vs Custom vs Naive on Real Cable Data ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Load real data
    trajectories = load_trajectories()
    
    # Run experiment
    results = run_experiment(trajectories)
    
    # Print results
    print_results(results)
    
    # Create plot
    create_plot(results)
    
    # Save
    np.save(OUTPUT_DIR / 'figure3_official_results.npy', results)
    print(f"\nResults saved to: {OUTPUT_DIR / 'figure3_official_results.npy'}")


if __name__ == "__main__":
    main()
