#!/usr/bin/env python3
"""
============================================================================
STEP 3: FAST Tokenization Comparison
============================================================================
This script compares:
    - FAST+ tokenizer from physical-intelligence/fast (HuggingFace)
    - Custom FAST tokenizer trained on berkeley_cable_routing 
    - OpenVLA ActionTokenizer (naive tokenizer)

============================================================================
"""

import argparse
import importlib.util
import json
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict

# Paths
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./results")
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_CUSTOM_FAST_SAVE_DIR = Path(
    os.environ.get("CUSTOM_FAST_SAVE_DIR", OUTPUT_DIR / "custom_fast_tokenizer")
)

# Chunking (match paper: 1-second chunks at 10 Hz)

CONTROL_FREQUENCY_HZ = 10
DEFAULT_CHUNK_SECONDS = 1.0
DEFAULT_CHUNK_STEPS = int(CONTROL_FREQUENCY_HZ * DEFAULT_CHUNK_SECONDS)

# ============================================================================
# FAST TOKENIZERS (official FAST+ and custom via .fit)
# ============================================================================

def load_official_fast_tokenizer():
    """Load the official FAST tokenizer from HuggingFace (FAST+ weights)."""
    try:
        from transformers import AutoProcessor
        
        print("Loading FAST+ tokenizer from physical-intelligence/fast...")
        tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast",
            trust_remote_code=True
        )
        print("Loaded successfully")
        return tokenizer
    
    except Exception as e:
        raise RuntimeError(
            "Failed to load FAST+ tokenizer from physical-intelligence/fast.\n"
            f"Error: {e}"
        )


def train_custom_fast_tokenizer(
    action_chunks: np.ndarray,
    *,
    vocab_size: int,
    scale: float,
    save_dir: Path,
):
    """Train a custom FAST tokenizer via .fit() on normalized action chunks."""
    tokenizer = load_official_fast_tokenizer()
    try:
        trained = tokenizer.fit(action_chunks, vocab_size=vocab_size, scale=scale)
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            trained.save_pretrained(save_dir)
            print(f"Saved custom FAST tokenizer to {save_dir}")
        except Exception as exc:
            print(f"WARNING: Failed to save custom FAST tokenizer: {exc}")
        return trained
    except Exception as exc:
        raise RuntimeError(
            "Failed to train custom FAST tokenizer with .fit(). "
            "Make sure the FAST implementation supports fit() in your environment.\n"
            f"Error: {exc}"
        )


def load_or_train_custom_fast_tokenizer(
    action_chunks: np.ndarray,
    *,
    vocab_size: int,
    scale: float,
    save_dir: Path,
    force_retrain: bool,
):
    """Load a saved custom FAST tokenizer if available, otherwise train + save it."""
    if save_dir.is_dir() and not force_retrain:
        try:
            from transformers import AutoProcessor, AutoTokenizer
            print(f"Loading custom FAST tokenizer from {save_dir}...")
            # Load the official processor code, then swap in the trained tokenizer files.
            processor = AutoProcessor.from_pretrained(
                "physical-intelligence/fast",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                save_dir,
                trust_remote_code=True,
            )
            if hasattr(processor, "bpe_tokenizer"):
                processor.bpe_tokenizer = tokenizer
            # Restore processor config fields if present
            config_path = save_dir / "processor_config.json"
            if config_path.is_file():
                with config_path.open("r") as f:
                    config = json.load(f)
                for key, value in config.items():
                    if hasattr(processor, key):
                        setattr(processor, key, value)
            print("Loaded successfully")
            return processor
        except Exception as exc:
            print(f"WARNING: Failed to load custom FAST tokenizer; retraining. Error: {exc}")
    return train_custom_fast_tokenizer(
        action_chunks,
        vocab_size=vocab_size,
        scale=scale,
        save_dir=save_dir,
    )

# ============================================================================
# NAIVE TOKENIZER (OpenVLA)
# ============================================================================

# Base tokenizer used by OpenVLA ActionTokenizer (tokenizer vocab size matters).
OPENVLA_TOKENIZER_ID = os.environ.get("OPENVLA_TOKENIZER_ID", "openvla/openvla-7b")


class OpenVLAActionTokenizerAdapter:
    """Adapter around OpenVLA's ActionTokenizer to return token counts."""

    def __init__(self, action_tokenizer):
        self.action_tokenizer = action_tokenizer
        self.min_action = float(action_tokenizer.min_action)
        self.max_action = float(action_tokenizer.max_action)
        self.bins = action_tokenizer.bins
        self.vocab_size = int(action_tokenizer.tokenizer.vocab_size)

    def count_tokens(self, action_chunk: np.ndarray) -> int:
        action = np.clip(action_chunk, a_min=self.min_action, a_max=self.max_action)
        discretized_action = np.digitize(action, self.bins)
        # Token ids are mapped to the tail of the vocab in OpenVLA
        _token_ids = self.vocab_size - discretized_action
        return int(_token_ids.size)


def load_openvla_action_tokenizer():
    """Load OpenVLA's official ActionTokenizer."""
    action_tokenizer_path = None
    for base in sys.path:
        candidate = Path(base) / "prismatic" / "vla" / "action_tokenizer.py"
        if candidate.is_file():
            action_tokenizer_path = candidate
            break
    if action_tokenizer_path is None:
        raise RuntimeError(
            "OpenVLA ActionTokenizer file not found. Install OpenVLA with:\n"
            "  uv sync\n"
            "or\n"
            "  uv pip install git+https://github.com/openvla/openvla"
        )
    spec = importlib.util.spec_from_file_location("openvla_action_tokenizer", action_tokenizer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load OpenVLA ActionTokenizer module spec.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ActionTokenizer = module.ActionTokenizer

    try:
        from transformers import AutoTokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(
            OPENVLA_TOKENIZER_ID,
            trust_remote_code=True,
        )
        return ActionTokenizer(base_tokenizer, bins=256, min_action=-1, max_action=1)
    except Exception as exc:
        raise RuntimeError(f"OpenVLA base tokenizer load failed: {exc}")


# ============================================================================
# EXPERIMENT
# ============================================================================

def load_trajectories():
    """Load saved trajectories (required)."""
    data_path = DATA_DIR / "cable_trajectories.npz"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}. Run step2_load_data.py first.")
    
    data = np.load(data_path, allow_pickle=True)
    trajectories = list(data['trajectories'])
    print(f"Loaded {len(trajectories)} trajectories from {data_path}")
    return trajectories


def run_comparison(
    trajectories: List[np.ndarray],
    *,
    chunk_steps: int,
    stride: int,
    vocab_size: int,
    scale: float,
    custom_save_dir: Path,
    force_retrain: bool,
):
    """Compare FAST+, custom FAST, and OpenVLA naive on fixed-length chunks."""
    
    print()
    print("=" * 60)
    print("TOKENIZATION COMPARISON")
    print("=" * 60)
    
    # Combine all actions for fitting
    all_actions = np.vstack(trajectories)
    print(f"Total actions: {len(all_actions)}")
    print(f"Action dimensions: {all_actions.shape[1]}")
    print(f"Chunk length: {chunk_steps} steps ({chunk_steps / CONTROL_FREQUENCY_HZ:.1f}s at {CONTROL_FREQUENCY_HZ} Hz)")
    print(f"Stride: {stride} steps")
    print(f"Custom FAST vocab size: {vocab_size} | scale: {scale}")
    
    # Load official FAST+ tokenizer (required)
    fast_plus_tokenizer = load_official_fast_tokenizer()
    using_official = True
    
    # Naive tokenizer (OpenVLA official)
    openvla_tokenizer = load_openvla_action_tokenizer()
    naive_tokenizer = OpenVLAActionTokenizerAdapter(openvla_tokenizer)
    use_openvla_naive = True
    
    # Normalize data for FAST (recommended: [-1, 1] range)
    q1 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)
    
    def normalize_chunk(chunk):
        range_vals = np.maximum(q99 - q1, 1e-8)
        return np.clip(2 * (chunk - q1) / range_vals - 1, -1, 1)
    
    # Build fixed-length chunks
    chunks: list[np.ndarray] = []
    for traj in trajectories:
        if len(traj) < chunk_steps:
            continue
        for start in range(0, len(traj) - chunk_steps + 1, stride):
            chunks.append(traj[start:start + chunk_steps])
    if not chunks:
        raise RuntimeError("No chunks available. Check chunk_steps/stride or dataset length.")

    # Normalize chunks for FAST
    chunks_norm = np.stack([normalize_chunk(chunk) for chunk in chunks])

    # Load or train custom FAST tokenizer on normalized Berkeley chunks
    custom_tokenizer = load_or_train_custom_fast_tokenizer(
        chunks_norm,
        vocab_size=vocab_size,
        scale=scale,
        save_dir=custom_save_dir,
        force_retrain=force_retrain,
    )

    # Compare tokenization
    results = {
        # FAST+ (official, required)
        'fast_tokens': [],
        # Custom FAST (trained via .fit)
        'custom_tokens': [],
        # Baseline
        'naive_tokens': [],
        'chunk_lengths': [],
    }
    
    print()
    print("Tokenizing trajectories...")
    
    total_chunks = 0
    for i, chunk in enumerate(chunks):
        chunk_norm = chunks_norm[i]

        # FAST+ tokenization (official expects normalized input)
        fast_result = fast_plus_tokenizer(chunk_norm[np.newaxis, ...])
        fast_token_count = len(fast_result[0]) if isinstance(fast_result, list) else len(fast_result)
        results['fast_tokens'].append(fast_token_count)

        # Custom FAST tokenization (trained tokenizer)
        custom_result = custom_tokenizer(chunk_norm[np.newaxis, ...])
        custom_token_count = len(custom_result[0]) if isinstance(custom_result, list) else len(custom_result)
        results['custom_tokens'].append(custom_token_count)

        # Naive tokenization (OpenVLA official)
        naive_count = naive_tokenizer.count_tokens(chunk)
        results['naive_tokens'].append(naive_count)
        results['chunk_lengths'].append(len(chunk))
        total_chunks += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i} chunks...")
    
    results['using_official_fast'] = using_official
    results['using_openvla_naive'] = use_openvla_naive
    results['chunk_steps'] = chunk_steps
    results['stride'] = stride
    results['chunk_seconds'] = chunk_steps / CONTROL_FREQUENCY_HZ
    results['total_chunks'] = total_chunks
    return results


def print_results(results: Dict):
    """Print formatted results."""
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    avg_naive = np.mean(results['naive_tokens'])
    avg_fast = np.mean(results['fast_tokens'])
    avg_custom = np.mean(results['custom_tokens'])
    avg_length = np.mean(results['chunk_lengths'])
    
    compression_fast = avg_naive / avg_fast
    compression_custom = avg_naive / avg_custom
    generalization_ratio = compression_fast / compression_custom if compression_custom > 0 else 0.0
    
    print()
    print(f"Using official FAST+ tokenizer: {results['using_official_fast']}")
    print(f"Using official OpenVLA naive tokenizer: {results.get('using_openvla_naive', False)}")
    print()
    print("TOKEN COUNTS (average per 1s chunk):")
    print(f"  Total chunks:                {results.get('total_chunks', 0)}")
    print(f"  Chunk length:                {avg_length:.1f} timesteps")
    print(f"  Naive tokenization:          {avg_naive:.1f} tokens")
    print(f"  FAST+ tokenization:          {avg_fast:.1f} tokens")
    print(f"  Custom FAST tokenization:    {avg_custom:.1f} tokens")
    print()
    print("COMPRESSION RATIOS (from average token counts):")
    print(f"  FAST+ compression:           {compression_fast:.2f}x  ← Primary metric")
    print(f"  Custom FAST compression:     {compression_custom:.2f}x  ← Primary metric")
    print(f"  Generalization ratio:        {generalization_ratio:.1%}")
    print()
    
    # Compare to paper's Table 1
    print("-" * 60)
    print("COMPARISON TO PAPER (Table 1)")
    print("-" * 60)
    print()
    print("  Paper reports for similar datasets:")
    print("    BridgeV2 (5Hz):     1.75x compression")
    print("    DROID (15Hz):       3.6x compression")
    print("    Bussing (20Hz):     5.0x compression")
    print("    Shirt Fold (50Hz):  13.2x compression")
    print()
    print(f"  Our FAST+ result:     {compression_fast:.1f}x compression")
    
    if compression_fast >= 2.0:
        print()
        print("  ✅ Our compression ratio is consistent with the paper!")
    
    return {
        'avg_naive_tokens': avg_naive,
        'avg_fast_tokens': avg_fast,
        'avg_custom_tokens': avg_custom,
        'compression_ratio': compression_fast,
        'compression_plus': compression_fast,
        'compression_custom': compression_custom,
        'generalization_ratio': generalization_ratio,
        'using_official': results['using_official_fast'],
    }


def main():
    parser = argparse.ArgumentParser(description="FAST tokenization comparison.")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--stride", type=int, default=DEFAULT_CHUNK_STEPS)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--custom-save-dir", type=str, default=str(DEFAULT_CUSTOM_FAST_SAVE_DIR))
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    chunk_steps = int(round(CONTROL_FREQUENCY_HZ * args.chunk_seconds))
    if chunk_steps <= 0:
        raise ValueError("chunk-seconds must be > 0")
    if args.stride <= 0:
        raise ValueError("stride must be > 0")

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " STEP 3: FAST Tokenization Comparison ".center(58) + "║")
    print("║" + " (Using Official physical-intelligence/fast) ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Load data
    trajectories = load_trajectories()
    
    # Run comparison
    results = run_comparison(
        trajectories,
        chunk_steps=chunk_steps,
        stride=args.stride,
        vocab_size=args.vocab_size,
        scale=args.scale,
        custom_save_dir=Path(args.custom_save_dir),
        force_retrain=args.force_retrain,
    )
    
    # Print results
    summary = print_results(results)
    
    # Save
    results_path = OUTPUT_DIR / "tokenization_results.npy"
    np.save(results_path, {**results, **summary})
    print()
    print(f"Results saved to {results_path}")
    
    print()
    print("=" * 60)
    print("NEXT STEP:")
    print("  python step3b_prediction_task.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
