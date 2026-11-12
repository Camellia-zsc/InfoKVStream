# StreamForest-inspired EventKV

基于 Qwen2.5-VL-7B 的流式视频分层记忆：保留 Recent 帧视觉向量，经短期窗口形成事件，在长期区合并事件；部分旧事件转为常驻 KV，并在独立 KV 超额时按 N/A/D 评分淘汰。高代价合并使用文字补偿。

本仓库提供两个明确分开的版本，共用必要的视频采样、Qwen 推理与评测工具。它不是官方 StreamForest 的模型，也不包含旧 InfoKV、HERMES 或矩摘要 EviM 的独立算法实现；共用基础代码的来源与许可仍予以保留。

## 两个版本

| 版本 | 代码 | 运行配置 | 主要行为 |
|---|---|---|---|
| 初步改进 | [versions/initial_improvement](versions/initial_improvement) | [configs/initial_improvement.yaml](configs/initial_improvement.yaml) | 长期超额：先合并，仍超额才转出最旧事件；秒时刻代价；摘要可见历史 KV |
| 最新确认 | [versions/latest](versions/latest) | [configs/latest.yaml](configs/latest.yaml) | 长期超额：先判断最老事件深度≥3或与其直接后继相似度<0.7；否则合并；采样帧时间代价；摘要只看相关事件 |

每轮只执行一次转出或合并，仍超额才再判断；预算满足后不额外转出。独立 KV 的 1000 是**每层常驻视觉 KV token 上限**，不包括文字、系统前缀、等待回答的视觉向量及临时回答 KV，不能当成模型总显存预算。

两版的完整差异、预算和历史结果见 [docs/VERSIONS.md](docs/VERSIONS.md)。初步改进对应已有 OVOBench 全量结果；最新组合配置尚无完整全量结果。此次发布仅完成 CPU 测试和数据预检，**没有使用整理后的入口重跑 7B 全量评测**。

## 安装

建议独立 Python 3.12 / CUDA 环境，先安装与 CUDA 匹配的 PyTorch 2.5.1，再安装本包：

```bash
pip install -e '.[test]'
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

评测配置默认 `flash_attention_2`、BF16、0.5 FPS、2 帧/chunk、最大图像边长 336。FA2 未安装时会报错，不会静默改用 SDPA。CPU 小模型测试使用 SDPA，是测试环境设置，不是正式评测配置。Qwen2.5-VL 依赖的缓存接口按 Transformers 4.57.3 固定。

模型默认 `Qwen/Qwen2.5-VL-7B-Instruct`，首次使用会由 Transformers 下载；已有权重时修改 YAML 的 `model_path` 为本地目录。仓库不包含权重、视频、标注、缓存或旧实验产物。

## 评测

复制并编辑数据配置中的路径。OVOBench 使用官方已经切好的视频（Forward 每个 test_info 对应一个单独视频），不在此工具中重切。

```bash
cp configs/ovobench.example.yaml ovo.local.yaml
# 编辑 annotation、chunked_dir
python -m eventkv.evaluate --config configs/initial_improvement.yaml \
  --benchmark-config ovo.local.yaml --output runs/initial_ovo --dry-run
python -m eventkv.evaluate --config configs/initial_improvement.yaml \
  --benchmark-config ovo.local.yaml --output runs/initial_ovo

python -m eventkv.evaluate --config configs/latest.yaml \
  --benchmark-config ovo.local.yaml --output runs/latest_bw30 \
  --tasks EPM ASI HLD --limit 30 --dry-run
```

`--limit 30` 只是选中任务中的前30问，**不保证每类10问，也不自动复现历史 OVOBW30 的固定清单**。精确复现小样本时应提供相同的标注子集，并保留其哈希；不要把任意前30问与旧结果直接比较。去掉 `--dry-run` 才会加载模型运行。

StreamingBench 使用 [configs/streamingbench.example.yaml](configs/streamingbench.example.yaml)，填写实际官方 CSV 和视频根目录；可按 `--tasks OP AP ...` 筛选。保持原实验 `full_prefix` 与官方问题模板、SQA 标注答案文字历史。此版本不支持 PO 主动输出；OSU 是纯视觉评测，没有音频。提供4类官方标注时，非PO全量应为4250问。

两 GPU 可用两个独立进程与不同输出目录（以下命令在两个终端执行）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m eventkv.evaluate --config configs/latest.yaml \
  --benchmark-config ovo.local.yaml --output runs/latest_ovo_s0 --num-shards 2 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -m eventkv.evaluate --config configs/latest.yaml \
  --benchmark-config ovo.local.yaml --output runs/latest_ovo_s1 --num-shards 2 --shard-index 1
```

同一命令再次运行会跳过已写入的问答。输出目录有独占锁；配置、代码、问题及提示内容、分片改变时拒绝混用旧结果。推理异常直接中止，保留已完成记录；空回答明确记为错误并计入分母，不静默排除。不要在相同输出目录中切换版本。

每问 JSONL 记录预测、原始回答、逐区预算/压缩统计、存活摘要内容、端到端耗时和 GPU 峰值；`memory.operation_log` 保留最近128次事件操作，`memory.management_log`（最新版）保存该问完整的超额管理决策与转出条件。耗时从视频读取到回答、统计快照和缓存清理，包含 CUDA 同步，不含模型加载和写文件。摘要文本是回答结束时的存活摘要，不保证记录所有已被删去的中间草稿。

`metrics.json` 提供分任务正确率与覆盖情况。OVO 三组宏平均及三组等权总平均，仅在对应任务全部出现时输出，是否全量完成仍须看 `complete`。本精简入口没有 Full KV 参考运行，**不产生 CC/CW/WC/WW、JS/KL、ΔP/ΔM 等配对 fidelity 指标**。

## Python 流式接口

```python
import yaml
from eventkv.model import EventKVModel

config = yaml.safe_load(open('configs/latest.yaml'))
model = EventKVModel(config)
try:
    model.reset()
    # frames 为已按 0.5 FPS 采样、最长边不超过336的 PIL.Image 列表，每块通常2帧。
    # model.stream_video_chunk(frames, seconds=4.0)
    # answer = model.ask_question('What happened earlier?', max_new_tokens=16)
finally:
    model.close()
```

同一流可多次询问；每次回答前封闭短期尾部，回答后释放临时后缀，常驻事件 KV 不丢失。新视频调用 `reset()`。

## 检查与来源

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q
```

测试覆盖真实小型 Qwen 的视觉编码/预填充/解码、缓存因果性、重复压缩只评分一次、原始存储释放、事件合并深度/时间代价、摘要上下文隔离与异常恢复、文字限额和独立评测协议。见 [验证说明](docs/VALIDATION.md)、[核心来源哈希](docs/source_provenance.json)、[第三方许可说明](THIRD_PARTY_NOTICES.md)。
