# WeightCreator v1

WeightCreator combines a frozen **Qwen3.5-0.8B** visual-language backbone, calibrated RGB-D geometry, and explicitly authored recurrent action experts for native LIBERO manipulation.

This first release packages the current internal **expert_v52** policy. The six expert checkpoints contain **1,603 scalar parameters**, excluding Qwen and the Python geometry/control infrastructure. The action weights are directly constructed and manually revised from development feedback; there is no gradient training, regression fitting, imitation learning, RL, or automatic weight search.

## Architecture

Instruction → explicit grammar/skill routing → RGB-D region proposals → frozen Qwen Yes/No logit margins → geometric relation selection → grasp/placement geometry → recurrent motor expert → native seven-dimensional action.

See [implementation details](docs/TECHNICAL.md), [evaluation protocol](docs/EVALUATION.md), and [release provenance](docs/RELEASE.md).

## Installation and execution

Use separate environments for model inference and simulation. The original deployment used Python 3.11 / Torch 2.6.0+cu124 for inference and Python 3.10 with native LIBERO, robosuite and MuJoCo for simulation. LIBERO and Qwen model files are external dependencies and are not redistributed here. Configure LIBERO's asset paths and an appropriate offscreen renderer for your installation.

```bash
export WEIGHTCREATOR_MODEL_DIR=/absolute/path/to/Qwen3.5-0.8B
export CUDA_VISIBLE_DEVICES=0
# Substitute your Python executables and a supported pick/place task.
SIM_PY=/absolute/path/to/simulation/python
MODEL_PY=/absolute/path/to/model/python
$SIM_PY capture_views.py --suite libero_object --task 0 --trial 0 --size 1024 --out outputs/example --split development
$SIM_PY perception.py outputs/example
$SIM_PY multiview_crops.py outputs/example
$MODEL_PY semantic_joint.py outputs/example --weights weights/expert_v52.pt --relations weights/relation_v7.pt
$SIM_PY rollout_program.py --suite libero_object --task 0 --trial 0 --capture outputs/example --out outputs/example/rollout --weights weights/expert_v52.pt --split development
```

For the complete four-suite pipeline, generate a protocol and invoke `worker.py`; workers share an output root and claim episodes via atomic directory creation:

```bash
python evaluation/make_protocol.py --out outputs/full --simulation-python "$SIM_PY" --model-python "$MODEL_PY"
python worker.py outputs/full/protocol.json outputs/full/producers/producer00
# Other processes use distinct producer directories and the same protocol.
```

Each process loads Qwen only in inference subprocesses. Set OMP/renderer thread limits to match allocated CPUs. The worker is bounded by the episode manifest and stage timeouts. Run workers on machines with the required simulation and CUDA environments; do not run the full workload on a memory-constrained login node.

## Results and limitations

The most recent fully audited historical cohort was **v36: 57/80 (71.25%)**. This is **not a v1/v52 score**. The current v1 full evaluation covers 40 tasks × 50 initial states = 2,000 episodes. Its final result is not claimed in this release.

The system uses depth and calibration, so results are RGB-D experiments, not directly comparable to RGB-only benchmark results. The full rerun includes development-used initial states. The explicit router supports 37 of 40 task instructions; unsupported instructions remain failures in the denominator. The requested 85% target has not yet been established.

No Qwen weights, simulator assets, demonstrations, credentials, cluster job credentials, or raw experiment directories are included.

Summarize saved outcomes without changing results:

```bash
python evaluation/summarize.py outputs/full/protocol.json
```

The summarizer withholds a final rate until all planned episodes are completed without infrastructure errors. This lightweight report does not certify platform terminal status.

## 中文说明

本项目正式命名为 **WeightCreator**，当前版本为 **第一版（v1.0.0）**，对应内部 `expert_v52`。

冻结的 Qwen 负责视觉语义匹配，RGB-D 几何模块生成抓取和放置条件，直接构造权重的循环动作专家输出七维机器人动作。权重通过人工分析与开发反馈修订，未使用梯度训练。指令解析、几何规则和原生控制器也是系统的重要组成部分。

本版发布核心代码、六组专家权重、可移植评测入口和技术文档，不包含 Qwen 模型文件或仿真资产。历史 57/80 成绩属于 v36；当前 v1 的 2,000 回合全量测评尚未在本发布中给出最终成绩，不能将历史成绩归属于本版，也不能宣称已达到 85%。
