"""Standalone streaming wrapper; no multi-method evaluation harness required."""
from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
import torch
from .video import iter_chunks_until_query, iter_frame_chunks, clean_reply, extract_option, normalize_forward_reply

VERSIONS = ("initial_improvement", "latest")

def configuration(config):
    version = config["version"]
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}")
    cls = import_module(f"versions.{version}.eventkv_config").EventKVConfig
    return cls.from_dict(config["eventkv"])

class EventKVModel:
    def __init__(self, config, *, qa_model=None, processor=None):
        self.config = config
        self.memory_config = configuration(config)
        inf = config["inference"]
        self.sample_fps = float(inf["sample_fps"])
        self.frames_per_turn = int(inf["frames_per_turn"])
        self.max_side = int(inf["max_side"])
        if self.sample_fps <= 0 or self.frames_per_turn < 1:
            raise ValueError("positive sampling rate and chunk length required")
        if qa_model is None:
            from .qwen_model import load_model
            qa_model, processor = load_model(config["model_path"], attention=inf["attn_implementation"], sample_fps=self.sample_fps)
        self.qa_model, self.qa_processor = qa_model, processor
        self.profiler = SimpleNamespace(phase=lambda *a, **k: nullcontext())
        cls = import_module(f"versions.{config['version']}.eventkv_gpu").EventKVGPUIntegrator
        with torch.inference_mode():
            self.controller = cls(self, SimpleNamespace(eventkv=self.memory_config))
        self.last_stats = {}
        self.last_summaries = []
        self.last_raw_reply = ""

    @torch.inference_mode()
    def reset(self):
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()
        self.controller.reset()

    @torch.inference_mode()
    def stream_video_chunk(self, frames, *, seconds=None):
        return self.controller.observe_chunk(frames, seconds=seconds)

    @torch.inference_mode()
    def ask_question(self, question, *, max_new_tokens=16):
        self.controller.prepare_answer()
        try:
            reply = self.qa_model.question_answering(
                {"question": question, "formatted_question": None, "prompt": self.qa_model.get_prompt(question)},
                max_new_tokens=max_new_tokens, temperature=0., repetition_penalty=1.)
            self.last_raw_reply = clean_reply(reply)
            return self.last_raw_reply
        finally:
            self.controller.finish_answer()

    def snapshot(self):
        self.last_stats = self.controller.get_stats()
        self.last_stats["operation_log"] = list(self.controller.memory.operation_log)
        self.last_stats["operation_log_limit"] = 128
        if hasattr(self.controller.memory, "management_log"):
            self.last_stats["management_log"] = list(self.controller.memory.management_log)
        tokenizer = self.qa_processor.tokenizer
        groups = (("long", self.controller.memory.long_events),
                  ("kv", self.controller.kv_events), ("text_only", self.controller.orphan_summaries))
        self.last_summaries = [dict(region=region, event_id=e.event_id, start=e.start_time,
            end=e.end_time, token_count=len(e.summary_ids), text=tokenizer.decode(e.summary_ids, skip_special_tokens=True))
            for region, events in groups for e in events if e.summary_ids]

    @torch.inference_mode()
    def infer(self, video, prompt, *, query_time=0., start_time=0., task=None):
        self.reset()
        self.last_raw_reply = ""
        try:
            if task in {"REC", "SSR", "CRR"}:
                count = 0
                for frames in iter_frame_chunks(video_path=video, sample_fps=self.sample_fps,
                        frames_per_turn=self.frames_per_turn, max_side=self.max_side):
                    self.stream_video_chunk(frames)
                    count += 1
                if not count:
                    raise RuntimeError(f"No video frames: {video}")
                raw = self.ask_question(prompt, max_new_tokens=64)
                return normalize_forward_reply(reply=raw, task=task)
            for frames, is_query, _ in iter_chunks_until_query(video_path=video,
                    query_time=float(query_time)-float(start_time), start_time=float(start_time),
                    sample_fps=self.sample_fps, frames_per_turn=self.frames_per_turn, max_side=self.max_side):
                self.stream_video_chunk(frames, seconds=len(frames)/self.sample_fps)
                if is_query:
                    return extract_option(self.ask_question(prompt, max_new_tokens=16))
            raise RuntimeError("No query turn was executed")
        finally:
            self.snapshot()
            self.qa_model.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def close(self):
        self.controller.close()
        self.qa_model.clear_cache()
