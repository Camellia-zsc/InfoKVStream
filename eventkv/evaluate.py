"""Standalone resumable evaluation of the two EventKV versions (no GPU at import)."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import yaml
from .adapters import OVOBenchAdapter, StreamingBenchAdapter, OVO_TASK_GROUPS, option_letter
from .prompts import build_prompt


def jobs_for(benchmark, config):
    """Each job resets video memory. SQA retains official ground-truth text history."""
    cls = OVOBenchAdapter if benchmark == "ovobench" else StreamingBenchAdapter
    for sample in cls(config).load_samples():
        history = []
        for q in sample.questions:
            path, start = sample.video_path, 0.
            if benchmark == "ovobench":
                if q.task_type in {"REC", "SSR", "CRR"}:
                    path = str(Path(config["chunked_dir"]) / f"{sample.sample_id}_{q.metadata['test_index']}.mp4")
                prompt = build_prompt(task=q.task_type, question=q.text, options=q.options,
                    _anno_=sample.metadata, index=q.metadata.get("test_index"))
            else:
                if q.response_policy == "standing_trigger":
                    raise ValueError("Proactive Output is not supported by this non-proactive release")
                history_text = ""
                if sample.metadata.get("category", "").lower().startswith("sequential"):
                    history_text = "Here are the contextual information related to the video. Please answer based on it:\n" + "\n".join(
                        f"At timestamp {old.metadata.get('time_stamp')}, Question: {old.text}; Answer: {answer};"
                        for old, answer in history) + "\n"
                prompt = history_text + f"Question: {q.text}\nOptions:\n" + "\n".join(q.options) + "\nPlease select the correct answer."
                if config.get("context_mode", "full_prefix") != "full_prefix":
                    raise ValueError("This release fixes StreamingBench to full_prefix, as in the EventKV runs")
                history.append((q, q.answer))
            yield dict(benchmark=benchmark, sample_id=sample.sample_id, question_id=q.question_id,
                task_type=q.task_type, task_code=q.metadata.get("task_code", q.task_type),
                video_path=path, query_time=float(q.timestamp or 0), start_time=start,
                ground_truth=q.answer, prompt=prompt)


def correct(row):
    response, gt = row.get("prediction"), row["ground_truth"]
    if response is None or row.get("status") == "error":
        return False
    response = str(response)
    if row["benchmark"] != "ovobench":
        return option_letter(response) == option_letter(gt)
    if row["task_type"] == "REC":
        return "".join(re.findall(r"\d+", response)) == str(gt)
    if row["task_type"] in {"SSR", "CRR"}:
        if (response == "N" and str(gt) == "0") or (response == "Y" and str(gt) == "1"):
            return True
        return ("No" if str(gt) == "0" else "Yes") in response
    return str(gt) in response


def metrics(rows, expected):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_type"]].append(correct(r))
    tasks = {k:dict(correct=sum(v), total=len(v), accuracy=sum(v)/len(v)) for k,v in by_task.items()}
    groups = {k:sum(tasks[t]["accuracy"] for t in ts)/len(ts)
              for k,ts in OVO_TASK_GROUPS.items() if all(t in tasks for t in ts)}
    return dict(completed=len(rows), expected=expected, complete=len(rows)==expected,
        errors=sum(r.get("status")=="error" for r in rows),
        accuracy=sum(correct(r) for r in rows)/len(rows) if rows else None,
        tasks=tasks, ovo_group_macro=groups,
        ovo_overall_macro=sum(groups.values())/3 if len(groups)==3 else None)


def source_hash():
    root=Path(__file__).resolve().parent.parent
    h=hashlib.sha256()
    for folder in ("eventkv", "versions"):
        for p in sorted((root/folder).rglob("*.py")):
            h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read_jsonl(path):
    if not path.exists(): return []
    # Fail explicitly on a torn line; never silently skip completed or corrupted work.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--benchmark-config", required=True)
    p.add_argument("--output", required=True, help="one directory per shard")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--limit", type=int, help="first N selected questions, before sharding")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args=p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("require 0 <= shard-index < num-shards")
    if args.limit is not None and args.limit < 1:p.error("limit must be positive")
    config=yaml.safe_load(Path(args.config).read_text())
    bench=yaml.safe_load(Path(args.benchmark_config).read_text())
    from .model import configuration
    configuration(config)
    benchmark=bench.pop("benchmark")
    if benchmark not in {"ovobench", "streamingbench"}:p.error("unknown benchmark")
    jobs=list(jobs_for(benchmark, bench))
    if args.tasks:jobs=[j for j in jobs if j["task_type"] in args.tasks or j["task_code"] in args.tasks]
    if args.limit:jobs=jobs[:args.limit]
    all_count=len(jobs)
    jobs=jobs[args.shard_index::args.num_shards]
    if not jobs:raise ValueError("No selected questions")
    ids=[j["question_id"] for j in jobs]
    if len(set(ids)) != len(ids):raise ValueError("Duplicate question IDs")
    missing=[j["video_path"] for j in jobs if not Path(j["video_path"]).is_file()]
    if missing:raise FileNotFoundError(f"{len(missing)} missing videos; first: {missing[0]}")
    identity=dict(model=config, benchmark=benchmark, benchmark_config=bench, source_hash=source_hash(),
        jobs_hash=hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(),
        shard_index=args.shard_index, num_shards=args.num_shards)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with (output/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract=output/"identity.json"
        if contract.exists() and json.loads(contract.read_text()) != identity:
            raise ValueError("Refusing resume with a changed source/config/dataset/selection")
        atomic_json(contract,identity)
        print(json.dumps(dict(total_selected=all_count, shard_questions=len(jobs),
            tasks=Counter(j["task_type"] for j in jobs), dry_run=args.dry_run),ensure_ascii=False))
        if args.dry_run:return
        import torch
        from .model import EventKVModel
        rows=read_jsonl(output/"predictions.jsonl")
        seen={r["question_id"] for r in rows}
        if len(seen)!=len(rows) or not seen.issubset(set(ids)):raise ValueError("Inconsistent resume rows")
        model=EventKVModel(config)
        try:
            for job in jobs:
                if job["question_id"] in seen:continue
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                row=dict(job)
                # Fail fast on inference errors; they are not converted into successful rows.
                row["prediction"]=model.infer(job["video_path"],job["prompt"],query_time=job["query_time"],
                    start_time=job["start_time"],task=job["task_type"])
                torch.cuda.synchronize()
                row.update(raw_output=model.last_raw_reply, elapsed_seconds=time.perf_counter()-started,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    memory=model.last_stats, summaries=model.last_summaries)
                if not str(row["prediction"]).strip():row["status"]="error"
                row["correct"]=correct(row)
                with (output/"predictions.jsonl").open("a") as f:
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
                rows.append(row)
                atomic_json(output/"metrics.json",metrics(rows,len(jobs)))
                print(f"{len(rows)}/{len(jobs)} {job['question_id']} correct={row['correct']}",flush=True)
        finally:model.close()
        atomic_json(output/"metrics.json",metrics(rows,len(jobs)))

if __name__ == "__main__":main()
