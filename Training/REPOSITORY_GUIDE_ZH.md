# LTN-SGG-RelTR：零基础中文阅读入口

本仓库是 RelTR（Relation Transformer）场景图生成器的实验扩展版。它保留“从一张
图中找出物体和物体关系”的神经网络主干，同时添加 LTN（Logic Tensor Networks）
神经符号约束：模型不仅要拟合标注，也被鼓励避免不符合常识/数据先验的三元组。

本文档体系假设读者没有接触过深度学习、目标检测、SGG、PyTorch 或模糊逻辑。请按
下列顺序阅读；每个专题均解释术语、代码入口、数学式、日志字段、常见误解和下一步。

## 固定比较基线

“标准 RelTR”指官方 [yrcong/RelTR](https://github.com/yrcong/RelTR) 的 `main` 分支，
本文档使用的固定提交为 `fca7397e9aaeccd95541e83afa4b971f3fa89014`
（2026-08-03 获取）。当前目录没有自己的 Git 历史；“新增”均是此基线与当前文件树
的静态差异，不将换行符、二进制扩展或未携带的 demo 文件误说成算法创新。

## 推荐阅读路线

1. [00_阅读路线与术语.md](docs/00_阅读路线与术语.md)：先建立“图像—框—类别—关系—训练”的整体心智模型。
2. [01_RelTR网络与Hungarian匹配.md](docs/01_RelTR网络与Hungarian匹配.md)：理解模型如何预测对象和三元组，以及预测为何必须与 GT 配对。
3. [02_损失函数与训练日志.md](docs/02_损失函数与训练日志.md)：重点阅读；完整解释 `loss_bbox`、`loss_giou`、`loss_ce`、`loss_rel`、`_0`–`_4`、LTN 损失及日志。
4. [03_LTN公理_约束与promptLLM.md](docs/03_LTN公理_约束与promptLLM.md)：理解神经符号扩展、所有约束文件和 LLM 生成链路。
5. [04_验证_COCO_SGG_输出文件与upload.md](docs/04_验证_COCO_SGG_输出文件与upload.md)：解释 validation、`eval.pth`、COCO evaluation accumulator、12 项 `coco_eval_bbox`、R/mR 与上传脚本。
6. [05_common_py参数手册.md](docs/05_common_py参数手册.md)：在你已经理解各模块后，逐参数配置实验。

## 快速运行前检查

- `--enable_ltn`、`--use_pair`、`--use_log`、`--use_hierarchy` 会改变损失、日志含义和
  所需约束文件；比较两个实验前必须核对它们。
- 非 pair 模式不要直接依赖当前 `--neg_constr_path` / `--pos_constr_path` 默认值：源码
  默认名和 `constraint/` 实际文件名不完全一致。请在命令中显式给出正、负 CSV。
- `checkpoint*.pth` 是可恢复训练/推理的模型快照；`eval.pth` 和 `eval/*.pth` **不是**
  模型快照，而是检测评估的中间/累积结果。详细区别见第 04 篇。
- `rel_error` 在部分 LTN 配置下会被调试值覆盖；判断谓词分类错误应看
  `rel_error_unscaled`。`-2.00` 表示 N/A（未启用、未触发或未计算），不是一个很差的
  分数。

## 仓库地图

| 路径 | 初学者应知道的职责 |
|---|---|
| `main.py` | 训练主入口：建模型/数据集、循环 epoch、保存 checkpoint、调用 validation。 |
| `engine.py` | 一个 epoch 的训练/评估循环；生成大多数终端日志字段。 |
| `models/reltr.py` | RelTR 网络、标准损失、LTN `SetCriterion` 和公理计算的核心。 |
| `models/fuzzy_logic.py` | 将神经网络输出转换为模糊逻辑谓词/掩码。 |
| `ltn/`, `ltn_log/` | 普通真值域与对数真值域的 LTN 运行时。 |
| `constraint/` | Visual Genome/LLM 形成的正负约束、domain/range、层次和关系难度表。 |
| `datasets/`, `lib/evaluation/` | 数据读取、COCO 检测评估和 SGG R/mR 评估。 |
| `util/common.py` | 所有命令行参数及少量实验辅助函数。 |
| `upload.py` | 把 Slurm/终端文本日志转为 CSV，并可上传 W&B；不训练模型。 |
| `promptLLM/`（项目根目录） | 离线生成/审计约束的脚本，不会被 `main.py` 自动执行。 |

## 使用方式

这些 Markdown 文件只增加说明，不改变模型、约束或日志。阅读时建议同时打开文档中
列出的源码函数；文档中“代码变量名”使用反引号，方便用全文搜索定位。
