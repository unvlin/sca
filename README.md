# SCA for TRL 0.24

This directory contains a standalone TRL implementation of **Segment-wise CoT Compression with Answer Alignment (SCA)**. It keeps the installed `trl` package untouched.

## Files

- `sca_trainer.py`: `SCATrainer`, a subclass of TRL's `GRPOTrainer`.
- `train_curriculum.py`: staged curriculum training entrypoint.
- `prepare_curriculum_by_stage.py`: converts staged curriculum source files into train/validation splits.

## Dataset requirement

Each example must contain:

```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "extra_info": {
        "ref_answer": "reference visible answer text"
    },
    "ground_truth": "final answer for correctness judging"
}
```

`extra_info["ref_answer"]` is required. Its token length is computed online with the training tokenizer for the answer-length reward.

## Method defaults

The trainer defaults match the paper method:

- Think compression margin: `m=256`
- Answer tolerance band: `f=32`
- Difficulty scale: `s=1.5`
- Training temperature schedule: cosine `1.3 -> 0.7`
- Answer alignment: gated forward KL on answer tokens as a separate loss

## Minimal usage

```python
from trl import GRPOConfig
from sca_trainer import SCAConfig, SCATrainer

def correctness_reward(prompts, completions, ground_truth, **kwargs):
    # Return > 0 for correct final answers, <= 0 otherwise.
    ...

args = GRPOConfig(
    output_dir="runs/sca",
    loss_type="dapo",
    beta=0.0,  # SCATrainer enables a reference model internally for answer alignment.
)

sca_config = SCAConfig(
    answer_tolerance_band=32.0,
    think_margin=256.0,
    diff_scale=1.5,
    answer_align_lambda=0.1,
    think_start_boundary="<think>",
    think_end_boundary="</think>",
    answer_start_boundary="",
    answer_end_boundary="<|im_end|>",
)

trainer = SCATrainer(
    model="Qwen/Qwen3-4B-Thinking-2507",
    reward_funcs=correctness_reward,
    args=args,
    sca_config=sca_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

## Curriculum training

```bash
PYTHONPATH=. python3 train_curriculum.py \
  --model /path/to/base-model \
  --data-root /path/to/sca/data/curriculum_by_stage \
  --eval-data-root /path/to/sca/data/curriculum_by_stage \
  --output-root /path/to/sca/runs/curriculum \
  --num-generations 32 \
  --answer-align-lambda 0.1 \
  --answer-tolerance-band 32 \
  --think-margin 256 \
  --difficulty-scale 1.5 \
  --temperature-start 1.3 \
  --temperature-end 0.7 \
  --deepspeed configs/ds_zero2.json \
  --gradient-checkpointing
```

Answer correctness is judged only by an OpenAI-compatible API. For local vLLM, start the judge model with `vllm serve ... --served-model-name your-vllm-served-model-name`; the script defaults to `http://127.0.0.1:8000/v1` and reads credentials from `JUDGER_API_KEY` unless an explicit API key is passed.
