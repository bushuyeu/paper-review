#!/bin/bash
#SBATCH --job-name=cosmos-r2
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=cosmos-r2-%j.out
#SBATCH --error=cosmos-r2-%j.err

set -e

DATA_DIR="/mnt/lustre/koa/scratch/$USER/koa-jobs"
RESULTS_DIR="${DATA_DIR}/results/${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

HF_CACHE="${DATA_DIR}/hf_cache"
mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN}"

VENV="/tmp/${USER}-cosmos-${SLURM_JOB_ID}"
module load lang/Python/3.11.5-GCCcore-13.2.0
python -m venv "${VENV}"
source "${VENV}/bin/activate"

pip install --retries 5 torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --retries 5 "transformers>=4.57.0" accelerate pillow

echo "=== GPU ==="
nvidia-smi

python << 'PYEOF'
import torch, json, os
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_id = "nvidia/Cosmos-Reason2-8B"
print(f"Loading {model_id}...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded!")

messages = [
    {"role": "system", "content": (
        "You are a robot planning assistant for a UR5e arm with a Robotiq Hand-E "
        "parallel gripper, an Axia80 force-torque sensor, and three wrist-mounted cameras. "
        "Answer in: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
    )},
    {"role": "user", "content": (
        "Plan a sequence of steps for the robot to pick up a flexible network cable "
        "by its RJ45 connector and insert it into a port on a server tray. "
        "Consider: the cable is deformable, the connector must be precisely aligned, "
        "and you have force-torque feedback available."
    )}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt").to(model.device)

print("Generating response...")
output = model.generate(**inputs, max_new_tokens=4096, temperature=0.6, top_p=0.95)
response = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print("=== RESPONSE ===")
print(response)

rd = os.environ["RESULTS_DIR"]
with open(f"{rd}/output.json", "w") as f:
    json.dump({"model": model_id, "response": response}, f, indent=2)
print(f"Saved to {rd}/output.json")
PYEOF

deactivate && rm -rf "${VENV}"
echo "Done! Results: ${RESULTS_DIR}"