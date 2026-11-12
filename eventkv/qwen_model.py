"""Native Qwen cache/greedy decode utilities. See THIRD_PARTY_NOTICES.md."""
import os
import re
import time
import logging
import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, DynamicCache
logger = logging.getLogger(__name__)

class EventKVQwen(Qwen2_5_VLForConditionalGeneration):
    def clear_cache(self):
        self.kv_cache = None
        self.conv_history = []
        self._position_ids_cache = [None] * self.num_layers
        self._layer_position_ids.clear()

    def get_prompt(self, query, mc=False):
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        return prompt + ("Best option: (" if mc else "")

    def _ensure_dynamic_cache(self):
        if self.kv_cache is None:
            return
        if not isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(self.kv_cache)

    def _append_position_ids_layer(self, layer_idx: int, start_per_dim: list, length: int):
        device = self.device
        new_pos = torch.zeros((3, length), device=device, dtype=torch.float32)
        for dim in range(3):
            new_pos[dim, :] = torch.arange(start_per_dim[dim], start_per_dim[dim] + length, device=device, dtype=torch.float32)
        if self._position_ids_cache[layer_idx] is None:
            self._position_ids_cache[layer_idx] = new_pos
        else:
            self._position_ids_cache[layer_idx] = torch.cat([self._position_ids_cache[layer_idx], new_pos], dim=1)

    def _append_position_ids_layer_explicit(self, layer_idx: int, new_pos: torch.Tensor):
        if self._position_ids_cache[layer_idx] is None:
            self._position_ids_cache[layer_idx] = new_pos.to(self.device)
        else:
            self._position_ids_cache[layer_idx] = torch.cat([self._position_ids_cache[layer_idx], new_pos.to(self.device)], dim=1)

    def _get_cache_seq_len_per_layer(self) -> list:
        if self.kv_cache is None:
            return [0] * self.num_layers
        lengths = []
        for layer_idx in range(len(self.kv_cache)):
            k_layer, v_layer = self.kv_cache[layer_idx]
            lengths.append(k_layer.shape[2])
        return lengths

    def _get_next_global_offset_per_layer(self) -> list:
        offsets = []
        for layer_idx in range(self.num_layers):
            if layer_idx < len(self._position_ids_cache) and self._position_ids_cache[layer_idx] is not None and (self._position_ids_cache[layer_idx].numel() > 0):
                cache = self._position_ids_cache[layer_idx]
                global_max = cache.max().item()
                offsets.append(int(global_max) + 1)
            else:
                offsets.append(0)
        return offsets

    def _build_position_ids_3d_for_text(self, global_offset: int, q_len: int, batch: int) -> torch.Tensor:
        """[3, batch, q_len] -- text tokens use identical position across all 3 dims."""
        pos_1d = torch.arange(global_offset, global_offset + q_len, device=self.device, dtype=torch.float32)
        pos_2d = pos_1d.unsqueeze(0).expand(batch, -1)
        pos_3d = pos_2d.unsqueeze(0).expand(3, -1, -1).clone()
        return pos_3d

    def _build_position_ids_3d_for_vision(self, grid_pos_ids: torch.Tensor, batch: int) -> torch.Tensor:
        """[3, batch, q_len] -- vision tokens use grid-based 3D positions."""
        pos_3d = grid_pos_ids.unsqueeze(1).expand(3, batch, -1).clone()
        return pos_3d

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)
        seq_len = self.init_prompt_ids.shape[-1]
        pos_1d = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        position_ids_3d = pos_1d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).clone()
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True, position_ids=position_ids_3d)
        self.kv_cache = output.past_key_values
        self.visual_start_idx = self.kv_cache[0][0].shape[2]
        self._ensure_dynamic_cache()
        self.total_processed_frames = 0
        curr_lens = self._get_cache_seq_len_per_layer()
        for layer_idx in range(self.num_layers):
            pos = torch.arange(curr_lens[layer_idx], device=self.device, dtype=torch.float32)
            self._position_ids_cache[layer_idx] = pos.unsqueeze(0).expand(3, -1).clone()

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, temperature=0, repetition_penalty=1.1, pseudo_forward=False, allowed_first_token_ids=None):
        device = self.device
        gen_debug = os.environ.get('EVENTKV_GEN_DEBUG', '0') == '1'
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []
        start_time = time.perf_counter()
        prompt = input_text['prompt']
        input_ids = self.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        if gen_debug and (not pseudo_forward):
            print('actual_prompt:', repr(prompt))
            print('prompt_token_length:', input_ids.shape[1])
            print('decoded_actual_prompt:', repr(self.processor.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)))
        self._ensure_dynamic_cache()
        self._ensure_dynamic_cache()
        past_lens_prefill = self._get_cache_seq_len_per_layer()
        global_offset_prefill = self._get_next_global_offset_per_layer()
        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]
        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(global_offset_prefill[layer_idx], q_len_prefill, batch)
            self._layer_position_ids[layer_idx] = position_ids_3d
        position_ids_3d = self._build_position_ids_3d_for_text(global_offset_prefill[0], q_len_prefill, batch)
        out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=self.kv_cache, position_ids=position_ids_3d)
        past_key_values = out.past_key_values
        logits = self.lm_head(out.last_hidden_state)
        if gen_debug and (not pseudo_forward):
            print('\n========== FINAL GENERATION DEBUG ==========')
            print('question:', repr(input_text.get('question')))
            print('formatted_question:', repr(input_text.get('formatted_question')))
            print('cache types:', type(self.kv_cache), type(past_key_values))
            print('same cache object:', self.kv_cache is past_key_values)
            print('eos_token_id:', self.processor.tokenizer.eos_token_id)
            print('pad_token_id:', self.processor.tokenizer.pad_token_id)
        for layer_idx in range(self.num_layers):
            offset = global_offset_prefill[layer_idx]
            self._append_position_ids_layer(layer_idx, [offset, offset, offset], q_len_prefill)
        self._layer_position_ids.clear()
        for step in range(max_new_tokens):
            last_token_logits = logits[0, -1, :]
            if step == 0 and allowed_first_token_ids is not None:
                allowed_ids = list(dict.fromkeys((int(token_id) for token_id in allowed_first_token_ids)))
                if not allowed_ids:
                    raise RuntimeError('allowed_first_token_ids is empty')
                constrained_logits = torch.full_like(last_token_logits, -float('inf'))
                constrained_logits[allowed_ids] = last_token_logits[allowed_ids]
                last_token_logits = constrained_logits
                print(f'EVENTKV first-token constraint: allowed_first_token_ids={allowed_ids}')
            if gen_debug and (not pseudo_forward) and (step < 8):
                diagnostic_logits = last_token_logits.float()
                print(f'[step {step}] nan={torch.isnan(diagnostic_logits).sum().item()}, posinf={torch.isposinf(diagnostic_logits).sum().item()}, neginf={torch.isneginf(diagnostic_logits).sum().item()}')
                top_values, top_ids = torch.topk(diagnostic_logits, k=10)
                top_items = []
                for token_id, value in zip(top_ids.tolist(), top_values.tolist()):
                    token_text = self.processor.tokenizer.decode([token_id], skip_special_tokens=False)
                    top_items.append((token_id, repr(token_text), round(value, 4)))
                print(f'[step {step}] top10:', top_items)
            if repetition_penalty != 1.0 and len(output_ids) > 0:
                for token_id in set(output_ids):
                    if last_token_logits[token_id] < 0:
                        last_token_logits[token_id] *= repetition_penalty
                    else:
                        last_token_logits[token_id] /= repetition_penalty
            if temperature == 0.0:
                _, indices = torch.topk(last_token_logits, 1)
                token = int(indices[0])
            else:
                scaled_logits = last_token_logits / temperature
                scaled_logits = torch.nan_to_num(scaled_logits, nan=-float('inf'), posinf=float('inf'), neginf=-float('inf'))
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    _, indices = torch.topk(last_token_logits, 1)
                    token = int(indices[0])
            output_ids.append(token)
            if gen_debug and (not pseudo_forward) and (step < 8):
                chosen_text = self.processor.tokenizer.decode([token], skip_special_tokens=False)
                print(f'[step {step}] chosen:', token, repr(chosen_text))
            if gen_debug and (not pseudo_forward) and (step == 0):
                end_time = time.perf_counter()
                print(f'TTFT: {end_time - start_time} seconds')
            if token in stop_token_ids:
                break
            curr_global_offset = self._get_next_global_offset_per_layer()
            self._layer_position_ids.clear()
            for layer_idx in range(self.num_layers):
                pos_step_3d = self._build_position_ids_3d_for_text(curr_global_offset[layer_idx], 1, 1)
                self._layer_position_ids[layer_idx] = pos_step_3d
            position_ids_3d = self._build_position_ids_3d_for_text(curr_global_offset[0], 1, 1)
            out = self.language_model(input_ids=torch.as_tensor([[token]], device=device), use_cache=True, past_key_values=past_key_values, position_ids=position_ids_3d)
            logits = self.lm_head(out.last_hidden_state)
            past_key_values = out.past_key_values
            for layer_idx in range(self.num_layers):
                offset = curr_global_offset[layer_idx]
                self._append_position_ids_layer(layer_idx, [offset, offset, offset], 1)
            self._layer_position_ids.clear()
        output = self.processor.tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)
        if allowed_first_token_ids is not None:
            print(f'EVENTKV raw_output: {output!r}')
        if not pseudo_forward:
            current_question = input_text['question']
            current_options = None
            formatted_question = input_text.get('formatted_question', None)
            if formatted_question:
                option_matches = re.findall('\\([A-Z]\\)\\s*(.+?)(?=\\n\\([A-Z]\\)|\\nThe best answer|\\n*$)', formatted_question, re.DOTALL)
                if option_matches:
                    current_options = [opt.strip() for opt in option_matches]
            self.conv_history.append((current_question, output, current_options))
            logger.info(f'Saved conversation to history. Total conversations: {len(self.conv_history)}')
        self._truncate_kv_cache(past_lens_prefill)
        for layer_idx in range(self.num_layers):
            if self._position_ids_cache[layer_idx] is not None and self._position_ids_cache[layer_idx].shape[1] > past_lens_prefill[layer_idx]:
                self._position_ids_cache[layer_idx] = self._position_ids_cache[layer_idx][:, :past_lens_prefill[layer_idx]].contiguous()
        new_lens = self._get_cache_seq_len_per_layer()
        print(f'Answering Cache lengths: min={min(new_lens)}, max={max(new_lens)}')
        torch.cuda.empty_cache()
        return output

    def _truncate_kv_cache(self, target_lengths):
        if self.kv_cache is None:
            return
        truncated_cache = []
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_cache):
            if isinstance(target_lengths, int):
                target_len = target_lengths
            else:
                target_len = target_lengths[layer_idx]
            truncated_k = k_cache[:, :, :target_len, :]
            truncated_v = v_cache[:, :, :target_len, :]
            truncated_cache.append((truncated_k, truncated_v))
        if isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(truncated_cache)
        else:
            self.kv_cache = truncated_cache

def load_model(model_path, *, attention="flash_attention_2", sample_fps=0.5, device="cuda"):
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=False)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map=device, attn_implementation=attention, dtype=torch.bfloat16)
    model = EventKVQwen.__new__(EventKVQwen)
    model.__dict__ = base.__dict__.copy()
    model.processor = processor
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    model.init_prompt_ids = processor.tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    model.num_layers = len(model.language_model.layers)
    model._position_ids_cache = [None] * model.num_layers
    model._layer_position_ids = {}
    model.conv_history = []
    model.kv_cache = None
    model.sample_fps = sample_fps
    model.streaming = True
    model.total_processed_frames = 0
    model.last_encoded_frames = 0
    model.eval()
    return model, processor
