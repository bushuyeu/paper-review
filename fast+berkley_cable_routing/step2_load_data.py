#!/usr/bin/env python3
"""
============================================================================
STEP 2: Load Cable Manipulation Data
============================================================================
This script loads the berkeley_cable_routing dataset and extracts trajectories.

Usage:
    python step2_load_data.py

Output:
    - cable_trajectories.npz (saved action trajectories)
    - Prints dataset statistics
============================================================================
"""

# --- Imports
import numpy as np
from pathlib import Path

# --- Output paths and dataset id
OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

REPO_ID = "lerobot/berkeley_cable_routing"
DOWNLOAD_VIDEOS = True
FORCE_LEROBOT = True
# Set to None to load all episodes, or an int to limit.
MAX_EPISODES = None

# --- Small helpers to normalize HF/LeRobot sample values
def _to_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value

def _extract_action(sample):
    action = sample.get("action")
    if action is None:
        raise KeyError("Missing 'action' in dataset sample.")
    return np.array(action)

def _extract_task(sample):
    return (
        sample.get("language_instruction")
        or sample.get("task")
        or sample.get("language")
        or "cable routing"
    )

# --- Core parser: group per-frame samples into episode trajectories
def _build_trajectories_from_dataset(hf_dataset, max_episodes: int | None):
    trajectories = []
    episode_tasks = []
    current_episode = None
    current_actions = []

    print("Extracting trajectories...")
    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]
        ep_idx = _to_scalar(sample.get("episode_index"))
        if ep_idx is None:
            raise KeyError("Missing 'episode_index' in dataset sample.")

        if ep_idx != current_episode:
            if current_actions:
                trajectories.append(np.array(current_actions))
            current_episode = ep_idx
            current_actions = []
            episode_tasks.append(_extract_task(sample))

            if max_episodes is not None and len(trajectories) >= max_episodes:
                break
            if len(trajectories) % 50 == 0:
                print(f"  Loaded {len(trajectories)} episodes...")

        current_actions.append(_extract_action(sample))

    if current_actions:
        trajectories.append(np.array(current_actions))

    print(f"  Loaded {len(trajectories)} episodes total")
    return trajectories, episode_tasks

# --- Primary loader: use LeRobot if available, otherwise fall back to HF parquet
def load_from_lerobot(max_episodes: int | None = 200):
    """Load real cable routing data from LeRobot Hub."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        print(f"Loading {REPO_ID} with LeRobot...")
        if DOWNLOAD_VIDEOS:
            print("(Videos enabled — this can be multiple GB and may hit rate limits)")
        else:
            print("(Videos disabled — smaller download)")
        print()
        
        episodes = list(range(max_episodes)) if max_episodes is not None else None
        dataset = LeRobotDataset(
            REPO_ID,
            episodes=episodes,
            download_videos=DOWNLOAD_VIDEOS,
        )

        # Important: use the underlying parquet dataset to avoid video decoding during iteration.
        # Videos are still downloaded (if enabled) and remain available on disk for later use.
        return _build_trajectories_from_dataset(dataset.hf_dataset, max_episodes)
        
    except ImportError:
        msg = "LeRobot not installed. Install with: pip install lerobot"
        if FORCE_LEROBOT:
            raise RuntimeError(msg)
        print(msg)
        print("\nFalling back to direct HF dataset loading (no videos)...\n")
        return load_from_hf_direct(max_episodes=max_episodes)
    except Exception as exc:
        msg = (
            f"LeRobot dataset loader failed ({type(exc).__name__}: {exc})\n"
            "This script is configured to require LeRobot (no HF fallback).\n"
            "If you see a version-compatibility error, try:\n"
            "  pip install 'lerobot==0.3.2'\n"
            "and rerun from a Python 3.12/3.11 venv."
        )
        if FORCE_LEROBOT:
            raise RuntimeError(msg)
        print(msg)
        print("\nFalling back to direct HF dataset loading (no videos)...\n")
        return load_from_hf_direct(max_episodes=max_episodes)

# --- Fallback loader: pull parquet data directly from HF (no videos)
def load_from_hf_direct(max_episodes: int | None = None):
    """Load data directly from Hugging Face dataset parquet files (no LeRobot metadata)."""
    try:
        from huggingface_hub import snapshot_download
        from lerobot.datasets.utils import load_nested_dataset
    except ImportError as exc:
        raise ImportError("Missing dependencies for direct HF loading. Install lerobot and huggingface_hub.") from exc

    print(f"Loading {REPO_ID} directly from Hugging Face (data only, no videos)...")
    print()

    snapshot_path = Path(
        snapshot_download(
            REPO_ID,
            repo_type="dataset",
            allow_patterns=["data/**"],
            ignore_patterns=["videos/**"],
        )
    )
    data_dir = snapshot_path / "data"
    episodes = list(range(max_episodes)) if max_episodes is not None else None
    hf_dataset = load_nested_dataset(data_dir, episodes=episodes)
    return _build_trajectories_from_dataset(hf_dataset, max_episodes)





# --- Analysis: summary statistics for sanity checking
def analyze_dataset(trajectories):
    """Print dataset statistics."""
    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    episode_lengths = [len(t) for t in trajectories]
    action_dims = trajectories[0].shape[1]
    total_frames = sum(episode_lengths)
    
    # Action statistics
    all_actions = np.vstack(trajectories)
    
    print(f"Episodes:           {len(trajectories)}")
    print(f"Total frames:       {total_frames}")
    print(f"Action dimensions:  {action_dims}")
    print(f"Episode lengths:    {min(episode_lengths)} - {max(episode_lengths)} steps")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Control frequency:  10 Hz (from dataset metadata)")
    print()
    print("Action statistics (per dimension):")
    print(f"  Min:  {all_actions.min(axis=0)}")
    print(f"  Max:  {all_actions.max(axis=0)}")
    print(f"  Mean: {all_actions.mean(axis=0)}")
    print(f"  Std:  {all_actions.std(axis=0)}")
    
    return {
        'n_episodes': len(trajectories),
        'total_frames': total_frames,
        'action_dims': action_dims,
        'episode_lengths': episode_lengths,
        'action_stats': {
            'min': all_actions.min(axis=0),
            'max': all_actions.max(axis=0),
            'mean': all_actions.mean(axis=0),
            'std': all_actions.std(axis=0),
        }
    }


# --- Persistence: save trajectories and stats for downstream steps
def save_trajectories(trajectories, filepath):
    """Save trajectories to numpy archive."""
    # Convert list of variable-length arrays to object array
    traj_array = np.array(trajectories, dtype=object)
    np.savez(filepath, trajectories=traj_array)
    print(f"Saved {len(trajectories)} trajectories to {filepath}")


# --- Entry point: orchestrates loading, stats, and saving
def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " STEP 2: Load Cable Manipulation Data ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Load data
    trajectories, tasks = load_from_lerobot(max_episodes=MAX_EPISODES)
    
    # Analyze
    stats = analyze_dataset(trajectories)
    
    # Save for next step
    output_path = OUTPUT_DIR / "cable_trajectories.npz"
    save_trajectories(trajectories, output_path)
    
    # Also save stats
    stats_path = OUTPUT_DIR / "dataset_stats.npy"
    np.save(stats_path, stats)
    
    print()
    print("=" * 60)
    print("NEXT STEP:")
    print("  python step3_fast_tokenization.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
