# FAST Empiricist Experiment

## Hypothesis

> "FAST+ universal tokenizer achieves ≥80% of the compression ratio of a custom FAST tokenizer on cable manipulation data."

## Quick Start (uv, two envs)

```bash
# Why two envs?
# OpenVLA (official ActionTokenizer) depends on `draccus==0.8.0`,
# while LeRobot depends on `draccus==0.10.0`, so they cannot live
# in the same environment.

# 1) LeRobot env (data loading only)
uv venv --python 3.11 .venv-lerobot
source .venv-lerobot/bin/activate
uv pip install -r requirements-lerobot.txt

# (Optional) login to avoid HF rate limits for large downloads
hf auth login

python step2_load_data.py          # Load cable data (LeRobot)
deactivate

# 2) OpenVLA env (tokenization + hypothesis)
uv venv --python 3.11 .venv-openvla
source .venv-openvla/bin/activate
uv sync                            # installs OpenVLA + transformers==4.40.1

python step3_fast_tokenization.py  # Metric 1: Compression ratio (official OpenVLA naive)
python step3b_prediction_task.py   # Metric 2: Temporal autocorrelation
python step4_hypothesis_test.py    # Test hypothesis
```

## Two Key Metrics (From Your Feedback)

### Metric 1: Compression Ratio (Table 1 style)
**Script:** `step3_fast_tokenization.py`

Compares token count between:
- **Naive tokenization** (per-dimension binning, like RT-2/OpenVLA)
- **FAST+ tokenization** (universal tokenizer from `physical-intelligence/fast`)
- **Custom FAST tokenization** (trained via `.fit()` on this dataset using the same official code)

Expected result: **3-10x compression** on cable manipulation data.

### Metric 2: Temporal Autocorrelation (Figure 3 insight)
**Script:** `step3b_prediction_task.py`

Shows WHY FAST helps learning:
- At high frequencies, naive tokens are ~99% correlated
- This means next-token prediction is trivial → poor learning signal
- FAST decorrelates tokens via DCT

| Frequency | Naive Autocorrelation | Problem |
|-----------|----------------------|---------|
| 5 Hz | 0.93 | High |
| 10 Hz | 0.98 | Very high |
| 50 Hz | **0.999** | Almost identical! |

## Official FAST Tokenizer

The experiment uses the **official FAST tokenizer** released by Physical Intelligence:

```python
from transformers import AutoProcessor

tokenizer = AutoProcessor.from_pretrained(
    "physical-intelligence/fast",
    trust_remote_code=True
)
tokens = tokenizer(action_chunk)

# Train a custom tokenizer on your dataset
custom_tokenizer = tokenizer.fit(action_chunks)
```

Both FAST+ and custom FAST are computed using the official implementation from HuggingFace (no local fallback).

## Files

```
fast_experiment/
├── README.md                      # This file
├── run_experiment.py              # Run all steps
├── step1_setup.sh                 # Install dependencies
├── step2_load_data.py             # Load berkeley_cable_routing
├── step3_fast_tokenization.py     # Metric 1: Compression ratio
├── step3b_prediction_task.py      # Metric 2: Autocorrelation
├── step4_hypothesis_test.py       # Hypothesis test
├── data/                          # Downloaded/generated data
└── results/                       # Experiment results
```

## Dataset

**Primary:** `lerobot/berkeley_cable_routing`
- 1,647 episodes of real cable routing
- Franka robot arm (7-DoF)
- 10 Hz control frequency
- Task: Route cable through clips

This is directly relevant to the Intrinsic AI challenge (cable assembly).

## References

- **FAST Paper:** https://arxiv.org/abs/2501.09747
- **Official Tokenizer:** https://huggingface.co/physical-intelligence/fast
- **OpenPI Repository:** https://github.com/Physical-Intelligence/openpi
- **Dataset:** https://huggingface.co/datasets/lerobot/berkeley_cable_routing
- **Intrinsic Challenge:** https://www.intrinsic.ai/events/ai-for-industry-challenge
