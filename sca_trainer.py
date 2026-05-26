from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from accelerate.utils import gather, gather_object
from trl.data_utils import is_conversational
from trl.extras.profiling import profiling_decorator
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import nanstd, pad


@dataclass
class SCAConfig:
    think_start_boundary: str = "<think>"
    think_end_boundary: str = "</think>"
    answer_start_boundary: str = ""
    answer_end_boundary: str = "<|im_end|>"
    answer_tolerance_band: float = 32.0
    think_margin: float = 256.0
    diff_scale: float = 1.5
    answer_align_lambda: float = 0.1
    temperature_start: float = 1.3
    temperature_end: float = 0.7


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return max(0, min(start, len(sequence)))
    max_start = len(sequence) - len(pattern)
    for index in range(max(0, start), max_start + 1):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def _find_all_subsequences(sequence: list[int], pattern: list[int], start: int = 0) -> list[int]:
    if not pattern:
        return [max(0, min(start, len(sequence)))]
    matches = []
    index = max(0, start)
    while index <= len(sequence) - len(pattern):
        found = _find_subsequence(sequence, pattern, index)
        if found < 0:
            break
        matches.append(found)
        index = found + max(1, len(pattern))
    return matches


def _len_align_reward(answer_len: float, ref_len: float, tolerance_band: float) -> float:
    ref_len = float(max(float(ref_len), 1.0))
    answer_len = float(answer_len)
    tolerance_band = float(tolerance_band)
    if answer_len < ref_len:
        return float(math.exp(-abs(answer_len - ref_len) / ref_len))
    len_cap = ref_len + tolerance_band
    if ref_len <= answer_len <= len_cap:
        return 1.0
    return float(math.exp(-abs(answer_len - len_cap) / len_cap))


def _grouped_advantage(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1).repeat_interleave(num_generations)
    std = grouped.std(dim=1).repeat_interleave(num_generations)
    return (rewards - mean) / (std + 1e-4)


class SCATrainer(GRPOTrainer):
    """TRL GRPOTrainer variant for Segment-wise CoT Compression with Answer Alignment.

    This subclass leaves the installed TRL package untouched. It assumes reward functions return
    positive values for correct answers and non-positive values for incorrect answers.
    """

    def _answer_alignment_is_active(self) -> bool:
        return float(self.sca_config.answer_align_lambda) != 0.0

    def __init__(
        self,
        *args,
        sca_config: SCAConfig | None = None,
        **kwargs,
    ):
        self.sca_config = sca_config or SCAConfig()
        trl_args = kwargs.get("args")
        if trl_args is not None and self._answer_alignment_is_active() and getattr(trl_args, "beta", 0.0) == 0.0:
            # TRL creates a frozen reference model only when beta is nonzero. The custom
            # SCA loss below uses that reference model, but does not use TRL's built-in KL loss.
            trl_args.beta = 1.0
        if trl_args is not None:
            trl_args.loss_type = "dapo"
        super().__init__(*args, **kwargs)
        if self.loss_type != "dapo":
            raise ValueError('SCATrainer only supports loss_type="dapo"')
        if self.use_liger_loss:
            raise NotImplementedError("SCATrainer does not support TRL liger loss because it uses token advantages.")

    def _token_len(self, text: str) -> int:
        return len(self.processing_class.encode(text or "", add_special_tokens=False))

    def _boundary_ids(self, boundary: str) -> list[int]:
        if not boundary:
            return []
        return self.processing_class.encode(boundary, add_special_tokens=False)

    def _skip_optional_newline(self, token_ids: list[int], index: int) -> int:
        newline_ids = self._boundary_ids("\n")
        if newline_ids and token_ids[index : index + len(newline_ids)] == newline_ids:
            return index + len(newline_ids)
        return index

    def _decode_token_slice(self, token_ids: list[int], start: int, end: int) -> str:
        return self.processing_class.decode(
            token_ids[max(0, start) : max(start, end)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _parse_completion(self, token_ids: list[int]) -> dict[str, Any]:
        cfg = self.sca_config
        think_start_ids = self._boundary_ids(cfg.think_start_boundary)
        think_end_ids = self._boundary_ids(cfg.think_end_boundary)
        answer_start_ids = self._boundary_ids(cfg.answer_start_boundary)
        answer_end_ids = self._boundary_ids(cfg.answer_end_boundary)

        fmt = True
        if think_start_ids:
            think_start_boundary_start = _find_subsequence(token_ids, think_start_ids)
            if think_start_boundary_start < 0:
                # Qwen3 thinking chat templates may place the opening <think> in the prompt.
                # In that case the completion starts directly with thinking content.
                think_start = 0
            elif think_start_boundary_start > 0:
                prefix_text = self._decode_token_slice(token_ids, 0, think_start_boundary_start)
                fmt = fmt and not prefix_text.strip()
                think_start = think_start_boundary_start + len(think_start_ids)
            else:
                think_start = len(think_start_ids)
        else:
            think_start = 0

        think_end_matches = _find_all_subsequences(token_ids, think_end_ids, think_start) if think_end_ids else []
        fmt = fmt and len(think_end_matches) == 1
        if think_end_matches:
            think_end = think_end_matches[0]
            cursor = think_end + len(think_end_ids)
        else:
            think_end = think_start
            cursor = len(token_ids)

        cursor = self._skip_optional_newline(token_ids, cursor)
        if answer_start_ids:
            has_answer_start = token_ids[cursor : cursor + len(answer_start_ids)] == answer_start_ids
            fmt = fmt and has_answer_start
            if has_answer_start:
                cursor += len(answer_start_ids)
                cursor = self._skip_optional_newline(token_ids, cursor)
        answer_start = cursor

        if answer_end_ids:
            answer_end_matches = _find_all_subsequences(token_ids, answer_end_ids, answer_start)
            answer_end = answer_end_matches[0] if answer_end_matches else len(token_ids)
            answer_end_tail_start = answer_end + len(answer_end_ids)
            answer_boundary_tail = (
                self._decode_token_slice(token_ids, answer_end_tail_start, len(token_ids))
                if answer_end_matches
                else ""
            )
            answer_boundary_is_final = bool(answer_end_matches) and not answer_boundary_tail.strip()
            fmt = fmt and len(answer_end_matches) == 1 and answer_boundary_is_final
        else:
            answer_end = len(token_ids)

        think_text = self._decode_token_slice(token_ids, think_start, think_end)
        answer_text = self._decode_token_slice(token_ids, answer_start, answer_end).rstrip()
        think_token_len = max(0, think_end - think_start)
        answer_token_len = max(0, answer_end - answer_start)
        fmt = fmt and bool(think_text.strip()) and bool(answer_text.strip())

        return {
            "fmt": float(fmt),
            "think_text": think_text,
            "answer_text": answer_text,
            "think_start_token": think_start,
            "think_token_len": think_token_len,
            "answer_start_token": answer_start,
            "answer_token_len": answer_token_len,
        }

    def _scheduled_temperature(self, mode: str) -> float:
        cfg = self.sca_config
        if mode != "train":
            return float(cfg.temperature_end)
        max_steps = max(1, int(getattr(self.args, "max_steps", 1) or 1))
        progress = min(1.0, max(0.0, float(self.state.global_step) / float(max_steps)))
        mix = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(cfg.temperature_end + (cfg.temperature_start - cfg.temperature_end) * mix)

    def _apply_temperature(self, mode: str) -> None:
        temperature = self._scheduled_temperature(mode)
        self.temperature = temperature
        if hasattr(self, "generation_config"):
            self.generation_config.temperature = temperature
        self._metrics[mode]["sampling_temperature"].append(temperature)

    def _compute_dss_token_advantages(
        self,
        inputs: list[dict[str, Any]],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completions_text: list[str],
        rewards_per_func: torch.Tensor,
        ref_per_token_logps: torch.Tensor | None,
        old_per_token_logps: torch.Tensor | None,
        sampling_per_token_logps: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        device = completion_mask.device
        cfg = self.sca_config
        valid_completion_ids = [
            row[mask.bool()].tolist() for row, mask in zip(completion_ids.detach().cpu(), completion_mask.detach().cpu())
        ]
        parsed_local = [self._parse_completion(ids) for ids in valid_completion_ids]
        parsed = gather_object(parsed_local)
        process_slice = slice(
            self.accelerator.process_index * len(parsed_local),
            (self.accelerator.process_index + 1) * len(parsed_local),
        )

        fmt = torch.tensor([p["fmt"] for p in parsed], dtype=torch.float32, device=device)
        answer_scores = rewards_per_func.nansum(dim=1).to(device)
        raw_answer_correct = (answer_scores > 0).float()
        gate = fmt * raw_answer_correct
        ans = gate

        ref_answer_lens_local = []
        for example in inputs:
            extra_info = example.get("extra_info")
            if not isinstance(extra_info, dict) or extra_info.get("ref_answer") is None:
                raise ValueError('SCATrainer requires every sample to contain extra_info["ref_answer"]')
            ref_answer_lens_local.append(float(self._token_len(str(extra_info["ref_answer"]))))
        ref_answer_lens = gather_object(ref_answer_lens_local)

        answer_lens = torch.tensor([p["answer_token_len"] for p in parsed], dtype=torch.float32, device=device)
        think_lens = torch.tensor([p["think_token_len"] for p in parsed], dtype=torch.float32, device=device)
        ref_lens = torch.tensor(ref_answer_lens, dtype=torch.float32, device=device)

        r_len = torch.tensor(
            [
                _len_align_reward(
                    float(answer_len),
                    float(ref_len),
                    cfg.answer_tolerance_band,
                )
                for answer_len, ref_len in zip(answer_lens, ref_lens)
            ],
            dtype=torch.float32,
            device=device,
        ) * gate

        r_eff = torch.zeros_like(think_lens)
        group_acc = torch.zeros_like(think_lens)
        for start in range(0, len(parsed), self.num_generations):
            end = start + self.num_generations
            group_gate = gate[start:end]
            group_think = think_lens[start:end]
            group_acc[start:end] = group_gate.mean()
            correct = (group_gate > 0.5) & (group_think > 0)
            if correct.any():
                if int(correct.sum().item()) > 2:
                    min_len = group_think[correct].min()
                    max_len = group_think[correct].max()
                else:
                    min_len = torch.ones((), dtype=torch.float32, device=device)
                    max_len = torch.ones((), dtype=torch.float32, device=device)
                denom = max_len - min_len
                raw = 1.0 - (group_think - min_len) / (denom + 1e-4)
                raw = torch.clamp(raw, min=0.0, max=1.0)
                raw = torch.where(group_think <= float(cfg.think_margin), torch.ones_like(raw), raw)
                raw = torch.where(denom <= 1e-4, torch.ones_like(raw), raw)
                r_eff[start:end] = torch.where(correct, raw, torch.zeros_like(raw))

        adv_think = _grouped_advantage(r_eff, self.num_generations)
        adv_answer = _grouped_advantage(r_len, self.num_generations)
        diff_weight = (2.0 - group_acc) * float(cfg.diff_scale)
        adv_think = torch.where(adv_think > 0, adv_think * diff_weight, adv_think)

        adv_think_local = adv_think[process_slice]
        adv_answer_local = adv_answer[process_slice]
        gate_local = gate[process_slice]
        token_advantages = torch.zeros_like(completion_mask, dtype=torch.float32)
        think_content_mask = torch.zeros_like(completion_mask, dtype=torch.bool)
        answer_content_mask = torch.zeros_like(completion_mask, dtype=torch.bool)
        for row, p in enumerate(parsed_local):
            think_start = max(0, min(int(p["think_start_token"]), completion_mask.size(1)))
            think_end = max(think_start, min(think_start + int(p["think_token_len"]), completion_mask.size(1)))
            answer_start = max(0, min(int(p["answer_start_token"]), completion_mask.size(1)))
            answer_end = max(answer_start, min(answer_start + int(p["answer_token_len"]), completion_mask.size(1)))
            token_advantages[row, think_start:think_end] += adv_think_local[row]
            token_advantages[row, answer_start:answer_end] += adv_answer_local[row]
            think_content_mask[row, think_start:think_end] = True
            answer_content_mask[row, answer_start:answer_end] = True

        answer_alignment_mask = answer_content_mask.float() * gate_local.unsqueeze(1)
        local_dss_reward_signal = (r_eff + r_len)[process_slice]
        global_dss_reward_signal = self.accelerator.gather(local_dss_reward_signal)
        reward_std = global_dss_reward_signal.view(-1, self.num_generations).std(dim=1).mean().reshape(1)

        token_advantages = token_advantages * completion_mask.float()
        correct_count = gate.sum().clamp(min=1.0)
        correct_avg_think_len = ((think_lens * gate).sum() / correct_count).reshape(1)
        correct_avg_answer_len = ((answer_lens * gate).sum() / correct_count).reshape(1)
        correct_answer_len_align = (r_len.sum() / correct_count).reshape(1)
        logs = {
            # Backward-compatible metric names used by older stage transition tooling.
            "dss_fmt": fmt,
            "dss_ans": ans,
            "dss_r_eff": r_eff,
            "dss_r_len": r_len,
            "dss_group_acc": group_acc,
            "dss_ref_answer_len": ref_lens,
            "dss_answer_len": answer_lens,
            "dss_correct_answer_len_align_mean": correct_answer_len_align,
            "dss_correct_avg_think_tokens": correct_avg_think_len,
            # Paper-facing minimal training logs.
            "format_rate": fmt,
            "answer_correct_rate": raw_answer_correct,
            "gate_pass_rate": gate,
            "avg_group_success_rate": group_acc,
            "avg_think_reward": r_eff,
            "avg_answer_len_reward": r_len,
            "avg_think_length": think_lens,
            "avg_correct_think_length": correct_avg_think_len,
            "avg_answer_length": answer_lens,
            "avg_correct_answer_length": correct_avg_answer_len,
            "reward_std": reward_std,
        }
        completion_token_mask = completion_mask.bool()
        entropy_masks = {
            "policy_entropy_correct_mask": completion_token_mask & (gate_local > 0.5).unsqueeze(1),
            "policy_entropy_think_mask": completion_token_mask & think_content_mask,
            "policy_entropy_answer_mask": completion_token_mask & answer_content_mask,
            "answer_alignment_mask": answer_alignment_mask,
        }
        return token_advantages, gate, logs, entropy_masks

    @profiling_decorator
    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        self._apply_temperature(mode)
        prompts = [x["prompt"] for x in inputs]
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_logps_list, forward_kwargs = self._generate(
            prompts, images
        )
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        num_images = [len(img_list) for img_list in images] if images is not None else None
        ref_per_token_logps = None
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if (
                self.args.gradient_accumulation_steps % generate_every != 0
                or (self.use_vllm and self.vllm_importance_sampling_correction)
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if is_conversational(inputs[0]):
            completions = []
            reward_completions_text = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for prompt, completion in zip(prompts, reward_completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        token_advantages, _, dss_logs, entropy_masks = self._compute_dss_token_advantages(
            inputs=inputs,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completions_text=completions_text,
            rewards_per_func=rewards_per_func,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            sampling_per_token_logps=sampling_per_token_logps,
        )

        for key, value in dss_logs.items():
            mean_value = value.float().mean()
            self._metrics[mode][key].append(self.accelerator.gather(mean_value).mean().item())
        self._metrics[mode]["sequence_token_advantage_std"].append(
            self.accelerator.gather(token_advantages.sum(dim=1)).view(-1, self.num_generations).std(dim=1).mean().item()
        )
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(gather_object(token_advantages.sum(dim=1).tolist()))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
            self._metrics[mode][f"rewards/{name}/mean"].append(torch.nanmean(rewards_per_func[:, i]).item())
            self._metrics[mode][f"rewards/{name}/std"].append(nanstd(rewards_per_func[:, i]).item())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": token_advantages,
            "num_items_in_batch": num_items_in_batch,
            **entropy_masks,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        output["ref_per_token_logps"] = ref_per_token_logps
        for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids"]:
            if key in forward_kwargs:
                output[key] = forward_kwargs[key]
        if images is not None:
            output["num_images"] = num_images
        return output

    def _get_completion_logits(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        forward_kwargs = {
            key: inputs.get(key)
            for key in ["pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes", "token_type_ids"]
            if inputs.get(key) is not None
        }
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **forward_kwargs)
        logits = outputs.logits
        if logits.size(1) >= logits_to_keep + 1:
            return logits[:, -logits_to_keep - 1 : -1, :]
        return logits[:, -logits_to_keep:, :]

    def _compute_answer_alignment_loss(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        logits_to_keep: int,
        inputs: dict[str, Any],
        mode: str,
    ) -> torch.Tensor:
        device = input_ids.device
        zero = torch.zeros((), dtype=torch.float32, device=device)
        if not self._answer_alignment_is_active():
            return zero

        answer_mask = inputs.get("answer_alignment_mask")
        if answer_mask is None:
            return zero
        answer_mask = answer_mask.to(device=device, dtype=torch.float32) * completion_mask.float()
        local_tokens = answer_mask.sum().reshape(1)
        if local_tokens.item() == 0:
            self._metrics[mode]["answer_align_kl_mean"].append(0.0)
            self._metrics[mode]["answer_align_loss"].append(0.0)
            return zero

        current_logits = self._get_completion_logits(model, input_ids, attention_mask, logits_to_keep, inputs)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self._get_completion_logits(
                    self.ref_model, input_ids, attention_mask, logits_to_keep, inputs
                )
        else:
            unwrapped_model = self.accelerator.unwrap_model(model)
            adapter_context = (
                unwrapped_model.disable_adapter()
                if hasattr(unwrapped_model, "disable_adapter")
                else nullcontext()
            )
            with torch.no_grad(), adapter_context:
                ref_logits = self._get_completion_logits(model, input_ids, attention_mask, logits_to_keep, inputs)

        current_log_probs = F.log_softmax(current_logits.float(), dim=-1)
        ref_log_probs = F.log_softmax(ref_logits.float(), dim=-1)
        token_kl = (ref_log_probs.exp() * (ref_log_probs - current_log_probs)).sum(dim=-1)
        masked_kl_sum = (token_kl * answer_mask).sum().reshape(1)

        global_tokens = self.accelerator.gather(local_tokens).sum().clamp(min=1.0)
        global_kl_sum = self.accelerator.gather(masked_kl_sum).sum()
        kl_mean = global_kl_sum / global_tokens

        normalizer = global_tokens / self.accelerator.num_processes
        loss = float(self.sca_config.answer_align_lambda) * masked_kl_sum.squeeze(0) / normalizer
        global_loss = float(self.sca_config.answer_align_lambda) * kl_mean
        self._metrics[mode]["answer_align_kl_mean"].append(kl_mean.detach().item())
        self._metrics[mode]["answer_align_loss"].append(global_loss.detach().item())
        return loss

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        entropy_mask = (
            self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
            if self.top_entropy_quantile < 1.0
            else None
        )
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError("importance_sampling_level must be 'token' or 'sequence'")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
        loss = (per_token_loss * completion_mask).sum() / normalizer

        mode = "train" if self.model.training else "eval"
        loss = loss + self._compute_answer_alignment_loss(
            model,
            input_ids,
            attention_mask,
            completion_mask,
            logits_to_keep,
            inputs,
            mode,
        )
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        completion_entropy_mask = completion_mask.float()
        entropy_sum = (entropies * completion_entropy_mask).sum().reshape(1)
        entropy_count = completion_entropy_mask.sum().reshape(1)
        global_entropy_sum = self.accelerator.gather(entropy_sum).sum()
        global_entropy_count = self.accelerator.gather(entropy_count).sum()
        if global_entropy_count.item() > 0:
            self._metrics[mode]["policy_entropy_mean"].append(
                (global_entropy_sum / global_entropy_count.clamp(min=1.0)).item()
            )
        for metric_name, mask_name in [
            ("policy_entropy_correct_mean", "policy_entropy_correct_mask"),
            ("policy_entropy_think_mean", "policy_entropy_think_mask"),
            ("policy_entropy_answer_mean", "policy_entropy_answer_mask"),
        ]:
            mask = inputs.get(mask_name)
            if mask is None:
                continue
            mask = mask.to(device=entropies.device, dtype=entropies.dtype)
            masked_entropy_sum = (entropies * mask).sum().reshape(1)
            masked_entropy_count = mask.sum().reshape(1)
            global_masked_entropy_sum = self.accelerator.gather(masked_entropy_sum).sum()
            global_masked_entropy_count = self.accelerator.gather(masked_entropy_count).sum()
            if global_masked_entropy_count.item() > 0:
                self._metrics[mode][metric_name].append(
                    (global_masked_entropy_sum / global_masked_entropy_count.clamp(min=1.0)).item()
                )
        is_clipped = (coef_1 < 1 - self.epsilon_low) | (coef_1 > 1 + self.epsilon_high)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather((is_clipped.float() * completion_mask).sum() / completion_token_count).mean().item()
        )
        return loss
