# 初步改进版

使用根目录 `configs/initial_improvement.yaml`，通过 `python -m eventkv.evaluate` 运行。

本目录的三个文件是独立的配置、视觉事件管理和GPU KV集成实现；只依赖根目录的必要共享工具，不导入latest。长期区超额时，先合并一次，仍超额再转出最旧事件；秒时刻代价，摘要使用现有历史KV上下文。对应已有OVOBench全量评测的机制。

完整安装、输入协议和历史结果边界见根目录README及 `docs/VERSIONS.md`。历史成绩并非本次精简包重新评测所得。
