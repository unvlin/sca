from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig

try:
    from .sca_trainer import SCAConfig, SCATrainer
except ImportError:
    from sca_trainer import SCAConfig, SCATrainer


class ArgsFileParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str):
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        return shlex.split(line)


class LocalRunLogger:
    def __init__(self, log_root: Path, run_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_main_process = int(os.getenv("RANK", "0")) == 0
        self.run_dir = log_root / f"{timestamp}_{run_name}"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.text_path = self.run_dir / "train.log"
        if self.is_main_process:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"sca.{timestamp}.{run_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        if not self.is_main_process:
            self.logger.addHandler(logging.NullHandler())
            return

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(self.text_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message: str) -> None:
        if not self.is_main_process:
            return
        self.logger.info(message)

    def metrics(self, event: str, payload: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        record = {"event": event, "time": datetime.now().isoformat(timespec="seconds"), **payload}
        record = {key: _json_safe(value) for key, value in record.items()}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def save_config(self, payload: dict[str, Any]) -> None:
        if not self.is_main_process:
            return
        config_path = self.run_dir / "run_config.json"
        record = {key: _json_safe(value) for key, value in payload.items()}
        for key in list(record):
            if "api_key" in key and not key.endswith("_env"):
                record[key] = "***" if record[key] else None
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class LocalMetricsCallback(TrainerCallback):
    def __init__(self, run_logger: LocalRunLogger, prefix: str, mode: str = "any"):
        self.run_logger = run_logger
        self.prefix = prefix
        self.mode = mode

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True) or not logs:
            return
        is_eval = any(str(key).startswith("eval_") for key in logs)
        if self.mode == "train" and is_eval:
            return
        if self.mode == "eval" and not is_eval:
            return
        payload = {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in logs.items()
            if isinstance(value, (int, float, str, bool))
        }
        payload["step"] = int(state.global_step)
        self.run_logger.metrics(self.prefix, payload)

        summary_keys = [
            "loss",
            "learning_rate",
            "clip_ratio",
            "reward_std",
            "format_rate",
            "answer_correct_rate",
            "gate_pass_rate",
            "avg_group_success_rate",
            "avg_think_reward",
            "avg_answer_len_reward",
            "answer_align_kl_mean",
            "answer_align_loss",
            "avg_think_length",
            "avg_correct_think_length",
            "avg_answer_length",
            "avg_correct_answer_length",
            "policy_entropy_mean",
            "policy_entropy_correct_mean",
            "policy_entropy_think_mean",
            "policy_entropy_answer_mean",
        ]
        summary = " | ".join(f"{key}={payload[key]}" for key in summary_keys if key in payload)
        if not summary:
            summary = " | ".join(f"{key}={value}" for key, value in payload.items() if key != "step")
        self.run_logger.info(f"[{self.prefix}] step={state.global_step} | {summary}")


def extract_answer_text(text: str, cfg: SCAConfig) -> str:
    text = str(text)
    if cfg.think_start_boundary:
        if cfg.think_start_boundary == "<think>":
            match = re.search(r"<think\b[^>]*>", text, flags=re.I)
            if match:
                text = text[match.end() :]
        elif cfg.think_start_boundary in text:
            text = text.split(cfg.think_start_boundary, 1)[1]
    if cfg.think_end_boundary and cfg.think_end_boundary in text:
        text = text.split(cfg.think_end_boundary, 1)[1]
    text = text[1:] if text.startswith("\n") else text
    if cfg.answer_start_boundary and text.startswith(cfg.answer_start_boundary):
        text = text[len(cfg.answer_start_boundary) :]
    text = text.strip()
    if cfg.answer_end_boundary and text.endswith(cfg.answer_end_boundary):
        text = text[: -len(cfg.answer_end_boundary)]
    return text.strip()


def parse_judge_bool(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    lowered = lowered.strip(" .,:;!?`'\"()[]{}")
    match = re.match(r"^(true|false|1|0)\b", lowered)
    if not match:
        return False
    return match.group(1) in ("true", "1")


def truncate_judge_text(text: str, max_chars: int, side: str = "tail") -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if side == "head":
        return text[:max_chars]
    return text[-max_chars:]


def truncate_judge_text_by_tokens(text: str, tokenizer, max_tokens: int, side: str = "tail") -> str:
    text = str(text or "").strip()
    if max_tokens <= 0 or not text:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    token_ids = token_ids[:max_tokens] if side == "head" else token_ids[-max_tokens:]
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()


def make_api_judge(args):
    api_key = args.judge_api_key or os.getenv(args.judge_api_key_env, "") or "EMPTY"
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("API judge requires the openai package: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=args.judge_base_url)
    model = args.judge_model
    if not model:
        try:
            models = client.models.list(timeout=args.judge_timeout_s)
            model = models.data[0].id
        except Exception as exc:
            raise RuntimeError("--judge-model is required when the judge API model cannot be auto-detected") from exc
    judge_tokenizer = None
    if args.judge_context_window > 0:
        tokenizer_source = args.judge_tokenizer_path or args.judge_model
        if not tokenizer_source:
            raise RuntimeError("--judge-context-window requires --judge-tokenizer-path or --judge-model")
        try:
            judge_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                "--judge-context-window requires a loadable judge tokenizer. "
                "Set --judge-tokenizer-path to the local judge model directory."
            ) from exc

    system = (
        "You are a strict mathematical final-answer equivalence judge.\n"
        "Compare the FINAL answers only. Ignore reasoning, formatting, and wording differences.\n"
        "Output exactly one token: true or false.\n"
        "No punctuation, no explanation, no extra text."
    )

    def build_user_text(pred: str, gold: str) -> str:
        return (
            "Gold final answer:\n"
            f"{gold}\n\n"
            "Predicted final answer:\n"
            f"{pred}\n\n"
            "Are these final answers mathematically equivalent? Output true or false:"
        )

    def count_request_tokens(user: str) -> int:
        if judge_tokenizer is None:
            return 0
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return len(
                judge_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            return len(judge_tokenizer.encode(f"{system}\n\n{user}", add_special_tokens=False))

    def build_user(pred: str, gold: str) -> str:
        pred = str(pred or "").strip()
        gold = str(gold or "").strip()
        if judge_tokenizer is not None:
            request_budget = max(
                1,
                int(args.judge_context_window)
                - int(args.judge_max_tokens)
                - int(args.judge_context_safety_tokens),
            )
            overhead_tokens = count_request_tokens(build_user_text("", ""))
            content_budget = max(1, request_budget - overhead_tokens)
            gold_budget = min(max(0, int(args.judge_gold_token_budget)), max(0, content_budget // 2))
            pred_budget = max(1, content_budget - gold_budget)
            original_pred = pred
            original_gold = gold
            pred = truncate_judge_text_by_tokens(original_pred, judge_tokenizer, pred_budget, args.judge_truncate_side)
            gold = truncate_judge_text_by_tokens(original_gold, judge_tokenizer, gold_budget, args.judge_truncate_side)
            user = build_user_text(pred, gold)
            while count_request_tokens(user) > request_budget and pred_budget > 1:
                overflow = count_request_tokens(user) - request_budget
                pred_budget = max(1, pred_budget - max(16, overflow))
                pred = truncate_judge_text_by_tokens(
                    original_pred,
                    judge_tokenizer,
                    pred_budget,
                    args.judge_truncate_side,
                )
                user = build_user_text(pred, gold)
            return user
        pred = truncate_judge_text(pred, args.judge_max_pred_chars, args.judge_truncate_side)
        gold = truncate_judge_text(gold, args.judge_max_gold_chars, args.judge_truncate_side)
        return build_user_text(pred, gold)

    def judge(pred: str, gold: str) -> bool:
        user = build_user(pred, gold)
        last_error = None
        for attempt in range(max(1, args.judge_retries)):
            try:
                if args.judge_endpoint == "chat":
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.0,
                        max_tokens=args.judge_max_tokens,
                        timeout=args.judge_timeout_s,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                else:
                    prompt = f"{system}\n\n{user}"
                    resp = client.completions.create(
                        model=model,
                        prompt=prompt,
                        temperature=0.0,
                        max_tokens=args.judge_max_tokens,
                        timeout=args.judge_timeout_s,
                    )
                    raw = (resp.choices[0].text or "").strip()
                return parse_judge_bool(raw)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, args.judge_retries):
                    time.sleep(1.0)
        raise RuntimeError(f"API judge failed after {args.judge_retries} retries: {last_error}")

    return judge


def make_correctness_reward(cfg: SCAConfig, api_judge, run_logger: LocalRunLogger | None = None):
    def correctness_reward(prompts, completions, ground_truth, **kwargs):
        local_records = []
        for local_index, (completion, gold) in enumerate(zip(completions, ground_truth)):
            if isinstance(completion, list):
                text = completion[0]["content"]
            else:
                text = completion
            answer = extract_answer_text(text, cfg)
            local_records.append(
                {
                    "local_index": local_index,
                    "answer_text": answer,
                    "ground_truth": gold,
                    "reward": 0.0,
                }
            )

        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            rewards = []
            for record in local_records:
                rewards.append(1.0 if api_judge(record["answer_text"], record["ground_truth"]) else 0.0)
            return rewards

        rank = torch.distributed.get_rank()
        gathered_records = _distributed_gather_object(local_records)
        world_size = torch.distributed.get_world_size()
        local_rewards = [0.0 for _ in local_records]
        judge_tasks = [
            (target_rank, record)
            for target_rank, shard in enumerate(gathered_records)
            for record in shard
        ]

        if rank == 0:
            total = len(judge_tasks)
            if run_logger is not None:
                run_logger.info(f"Training reward: rank 0 starts API judging for {total} completions.")

        correct = 0
        for position, (target_rank, record) in enumerate(judge_tasks, start=1):
            payload = None
            if rank == 0:
                try:
                    is_correct = bool(api_judge(record["answer_text"], record["ground_truth"]))
                    reward = 1.0 if is_correct else 0.0
                    correct += int(is_correct)
                    payload = {
                        "ok": True,
                        "target_rank": target_rank,
                        "local_index": record["local_index"],
                        "reward": reward,
                    }
                except Exception as exc:
                    payload = {
                        "ok": False,
                        "error": f"API judge failed during training reward at item {position}: {exc}",
                    }
            payload = _distributed_broadcast_object(payload, src=0)
            if not payload["ok"]:
                raise RuntimeError(payload["error"])
            if payload["target_rank"] == rank:
                local_rewards[payload["local_index"]] = float(payload["reward"])

        if rank == 0 and run_logger is not None:
            run_logger.info(f"Training reward: API judging complete, correct={correct}/{len(judge_tasks)}.")

        if len(local_rewards) != len(local_records):
            raise RuntimeError(
                f"Training reward length mismatch on rank {rank}: "
                f"expected {len(local_records)}, got {len(local_rewards)}; world_size={world_size}"
            )
        return local_rewards

    return correctness_reward


def load_split_dataset(stage_dir: Path, split_name: str):
    jsonl_path = stage_dir / f"{split_name}.jsonl"
    parquet_path = stage_dir / f"{split_name}.parquet"
    if jsonl_path.exists():
        return load_dataset("json", data_files=str(jsonl_path), split="train")
    if parquet_path.exists():
        return load_dataset("parquet", data_files=str(parquet_path), split="train")
    raise FileNotFoundError(f"Expected {jsonl_path} or {parquet_path}")


def resolve_eval_stage_dir(train_stage_dir: Path, eval_data_root: Path | None) -> Path:
    if eval_data_root is None:
        return train_stage_dir
    stage_eval_dir = eval_data_root / train_stage_dir.name
    if stage_eval_dir.is_dir():
        return stage_eval_dir
    if eval_data_root.is_dir():
        return eval_data_root
    raise FileNotFoundError(f"Evaluation data root does not exist: {eval_data_root}")


def load_stage_dataset(stage_dir: Path, eval_data_root: Path | None = None, eval_split_name: str = "val"):
    train_dataset = load_split_dataset(stage_dir, "train")
    eval_stage_dir = resolve_eval_stage_dir(stage_dir, eval_data_root)
    eval_dataset = load_split_dataset(eval_stage_dir, eval_split_name)
    return train_dataset, eval_dataset, eval_stage_dir


def _distributed_broadcast_object(obj: Any, src: int = 0) -> Any:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload = [obj]
        torch.distributed.broadcast_object_list(payload, src=src)
        return payload[0]
    return obj


def _tokenizer_supports_chat_template_kwarg(tokenizer, kwarg_name: str) -> bool:
    try:
        messages = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **{kwarg_name: True},
        )
        return True
    except TypeError:
        return False
    except Exception:
        return True


def log_tokenizer_format_context(tokenizer, cfg: SCAConfig, run_logger: LocalRunLogger) -> None:
    special_map = getattr(tokenizer, "special_tokens_map", {})
    added_decoder = getattr(tokenizer, "added_tokens_decoder", {})
    run_logger.info(f"tokenizer special_tokens_map={special_map}")
    run_logger.info(f"tokenizer eos_token={getattr(tokenizer, 'eos_token', None)} eos_token_id={getattr(tokenizer, 'eos_token_id', None)}")
    run_logger.info(f"tokenizer pad_token={getattr(tokenizer, 'pad_token', None)} pad_token_id={getattr(tokenizer, 'pad_token_id', None)}")
    for boundary in [
        cfg.think_start_boundary,
        cfg.think_end_boundary,
        cfg.answer_start_boundary,
        cfg.answer_end_boundary,
        "\n",
    ]:
        if boundary:
            ids = tokenizer.encode(boundary, add_special_tokens=False)
            decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            token_info = [str(added_decoder.get(token_id, "")) for token_id in ids]
            run_logger.info(f"boundary={boundary!r} ids={ids} decoded={decoded!r} added_tokens={token_info}")
    supports_enable_thinking = _tokenizer_supports_chat_template_kwarg(tokenizer, "enable_thinking")
    run_logger.info(f"tokenizer chat_template supports enable_thinking={supports_enable_thinking}")


def make_training_args(args, output_dir: Path, max_steps: int, use_deepspeed: bool = True) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        remove_unused_columns=False,
        loss_type="dapo",
        beta=0.0,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        steps_per_generation=args.steps_per_generation,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=args.deepspeed if use_deepspeed else None,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def build_trainer(
    model_path: str,
    train_dataset,
    eval_dataset,
    tokenizer,
    sca_config,
    reward_func,
    train_args,
    callbacks=None,
):
    trainer = SCATrainer(
        model=model_path,
        reward_funcs=reward_func,
        args=train_args,
        sca_config=sca_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    for callback in callbacks or []:
        trainer.add_callback(callback)
    return trainer


def main() -> None:
    parser = ArgsFileParser(description="Sequential curriculum training for SCA.", fromfile_prefix_chars="@")
    parser.add_argument("--model", required=True, help="Initial model path or HF model id.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/curriculum_by_stage"),
    )
    parser.add_argument(
        "--eval-data-root",
        type=Path,
        default=None,
        help=(
            "Optional validation-data root. If it contains per-stage subdirectories, "
            "the script loads <eval-data-root>/<stage-name>/<eval-split-name>; "
            "otherwise it loads <eval-data-root>/<eval-split-name>."
        ),
    )
    parser.add_argument(
        "--eval-split-name",
        type=str,
        default="val",
        help="Validation split basename, e.g. val for val.jsonl/val.parquet.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/curriculum"))
    parser.add_argument("--log-root", type=Path, default=None, help="Directory for local SCA text and JSONL logs.")
    parser.add_argument(
        "--start-stage",
        type=str,
        default=None,
        help="Stage directory name to start from, e.g. stage_02_gsmk_deepmath_1_3_5. Earlier stages are skipped.",
    )
    parser.add_argument("--stage-steps", type=int, default=100)
    parser.add_argument("--answer-align-lambda", type=float, default=0.1)
    parser.add_argument("--answer-tolerance-band", type=float, default=32.0)
    parser.add_argument("--think-margin", type=float, default=256.0)
    parser.add_argument("--difficulty-scale", type=float, default=1.5)
    parser.add_argument("--think-start-boundary", type=str, default="<think>")
    parser.add_argument("--think-end-boundary", type=str, default="</think>")
    parser.add_argument("--answer-start-boundary", type=str, default="")
    parser.add_argument("--answer-end-boundary", type=str, default="<|im_end|>")
    parser.add_argument("--num-generations", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=32768)
    parser.add_argument("--eval-max-completion-length", type=int, default=None)
    parser.add_argument("--steps-per-generation", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to a DeepSpeed config JSON file.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7, help="Evaluation temperature and fallback fixed temperature.")
    parser.add_argument("--temperature-start", type=float, default=1.3)
    parser.add_argument("--temperature-end", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--judge-api-key", type=str, default=None, help="Judge API key. For local vLLM, omit this and EMPTY is used.")
    parser.add_argument("--judge-api-key-env", type=str, default="JUDGER_API_KEY")
    parser.add_argument("--judge-base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model name served by the OpenAI-compatible API.")
    parser.add_argument("--judge-endpoint", choices=["chat", "completion"], default="chat")
    parser.add_argument("--judge-timeout-s", type=int, default=60)
    parser.add_argument("--judge-max-tokens", type=int, default=4)
    parser.add_argument("--judge-tokenizer-path", type=str, default=None)
    parser.add_argument("--judge-context-window", type=int, default=0)
    parser.add_argument("--judge-context-safety-tokens", type=int, default=128)
    parser.add_argument("--judge-gold-token-budget", type=int, default=512)
    parser.add_argument("--judge-max-pred-chars", type=int, default=3000)
    parser.add_argument("--judge-max-gold-chars", type=int, default=1000)
    parser.add_argument("--judge-truncate-side", choices=["head", "tail"], default="tail")
    parser.add_argument("--judge-retries", type=int, default=3)
    args = parser.parse_args()
    state = PartialState()
    if args.eval_max_completion_length is None:
        args.eval_max_completion_length = args.max_completion_length

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.log_root or (args.output_root / "logs")
    run_logger = LocalRunLogger(log_root, run_name=Path(str(args.output_root)).name or "run")
    run_logger.info("Starting SCA curriculum training.")
    run_logger.info(f"Run log directory: {run_logger.run_dir}")
    run_logger.info(f"Model path: {args.model}")
    run_logger.info(f"Training data directory: {args.data_root}")
    run_logger.info(f"Validation data directory: {args.eval_data_root or args.data_root} split={args.eval_split_name}")
    run_logger.info(f"Output directory: {args.output_root}")
    run_logger.info(f"Training generation limit: max_completion_length={args.max_completion_length}")
    run_logger.info(f"Validation generation limit: eval_max_completion_length={args.eval_max_completion_length}")
    run_logger.info(f"Training metrics are logged every {args.logging_steps} step(s); WandB/TensorBoard are disabled.")
    run_logger.info("Saving the run configuration snapshot.")
    run_logger.save_config(vars(args))

    run_logger.info("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    run_logger.info("Tokenizer loaded.")

    run_logger.info("Initializing SCA configuration.")
    sca_config = SCAConfig(
        answer_align_lambda=args.answer_align_lambda,
        answer_tolerance_band=args.answer_tolerance_band,
        think_margin=args.think_margin,
        diff_scale=args.difficulty_scale,
        temperature_start=args.temperature_start,
        temperature_end=args.temperature_end,
        think_start_boundary=args.think_start_boundary,
        think_end_boundary=args.think_end_boundary,
        answer_start_boundary=args.answer_start_boundary,
        answer_end_boundary=args.answer_end_boundary,
    )
    run_logger.info(
        "SCA configuration initialized: "
        f"answer_align_lambda={args.answer_align_lambda}, "
        f"answer_tolerance_band={args.answer_tolerance_band}, "
        f"think_margin={args.think_margin}, "
        f"difficulty_scale={args.difficulty_scale}, "
        f"temperature=cosine:{args.temperature_start}->{args.temperature_end}"
    )
    log_tokenizer_format_context(tokenizer, sca_config, run_logger)

    run_logger.info("Initializing the API-judge correctness reward.")
    api_judge = make_api_judge(args)
    reward_func = make_correctness_reward(sca_config, api_judge, run_logger=run_logger)
    run_logger.info(
        f"API judge initialized: base_url={args.judge_base_url}, "
        f"model={args.judge_model or 'auto'}, endpoint={args.judge_endpoint}"
    )
    current_model = args.model

    stage_dirs = sorted(path for path in args.data_root.glob("stage_*") if path.is_dir())
    if not stage_dirs:
        run_logger.info(f"No stage_* directories found under {args.data_root}; stopping training.")
        raise FileNotFoundError(f"No stage_* directories found under {args.data_root}")
    if args.start_stage is not None:
        stage_names = [path.name for path in stage_dirs]
        if args.start_stage not in stage_names:
            raise ValueError(f"--start-stage={args.start_stage!r} does not exist; available stages: {stage_names}")
        start_index = stage_names.index(args.start_stage)
        skipped = stage_names[:start_index]
        stage_dirs = stage_dirs[start_index:]
        if skipped:
            run_logger.info(f"Skipped stages before --start-stage={args.start_stage}: {', '.join(skipped)}")
    run_logger.info(f"Found {len(stage_dirs)} curriculum stages: {', '.join(path.name for path in stage_dirs)}")

    for stage_dir in stage_dirs:
        run_logger.info(f"Loading training and validation datasets for {stage_dir.name}.")
        train_dataset, eval_dataset, eval_stage_dir = load_stage_dataset(
            stage_dir,
            eval_data_root=args.eval_data_root,
            eval_split_name=args.eval_split_name,
        )
        stage_output = args.output_root / stage_dir.name
        stage_output.mkdir(parents=True, exist_ok=True)
        run_logger.info(f"Entering stage: {stage_dir.name}")
        run_logger.info(
            f"{stage_dir.name} data loaded: train_size={len(train_dataset)} "
            f"eval_size={len(eval_dataset)} eval_dir={eval_stage_dir} output={stage_output}"
        )

        run_logger.info(f"{stage_dir.name}: building training arguments; evaluation runs on this stage after every step.")
        train_args = make_training_args(args, stage_output, max_steps=args.stage_steps)
        run_logger.info(
            f"{stage_dir.name}: ready to train, "
            f"max_steps={args.stage_steps}, model={current_model}, output={stage_output}"
        )
        trainer = build_trainer(
            current_model,
            train_dataset,
            eval_dataset,
            tokenizer,
            sca_config,
            reward_func,
            train_args,
            callbacks=[
                LocalMetricsCallback(run_logger, f"{stage_dir.name}/train", mode="train"),
                LocalMetricsCallback(run_logger, f"{stage_dir.name}/eval", mode="eval"),
            ],
        )
        run_logger.info(f"{stage_dir.name}: training started.")
        trainer.train()
        run_logger.info(f"{stage_dir.name}: training complete.")
        final_dir = stage_output / "final"
        run_logger.info(f"{stage_dir.name}: saving model and tokenizer.")
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        current_model = str(final_dir)
        run_logger.info(f"{stage_dir.name}: model saved to {final_dir}.")
        trainer.accelerator.free_memory()
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state.wait_for_everyone()

    run_logger.info(f"All curriculum stages complete. Final model path: {current_model}")


if __name__ == "__main__":
    main()
