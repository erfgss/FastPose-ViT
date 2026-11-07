# FastPose-ViT 文献驱动实验报告

已完成主实验 5 配置 × 3 随机种子 × 8 轮，以及3次有效权重衰减补充对照，共18次冻结分支训练，以及5种SPEED测试条件、SPEED+两个真实域和NPU延迟测试。验证集选择配置为 **disentangled**，代表模型种子预先固定123。配置/种子/测试条件均在测试前写入 protocol.json。

## 验证集结果（越低越好）

原始基线验证 SPEED：0.011002960。每次选择 epoch0–8 的最优验证权重；表中标准差是3个随机种子间的样本标准差，不能当作严格显著性检验。

| 配置 | 平均SPEED | 种子标准差 | 相对基线变化% |
|---|---|---|---|
| full | 0.0109501 | 0.0000021 | -0.4808085 |
| no_geometry | 0.0109602 | 0.0000014 | -0.3887308 |
| no_gates | 0.0109528 | 0.0000021 | -0.4562096 |
| disentangled | 0.0109486 | 0.0000022 | -0.4940480 |
| no_weight_decay | 0.0109501 | 0.0000021 | -0.4808085 |

完整双 Token 平均相对变化 -0.481%；选定配置平均相对变化 -0.494%。不要把所有收益归因于几何或门控，应分别比较 full 与对应消融。三个种子只提供初步稳定性证据。冻结骨干/回归头保留原基线能力，因此整体接近原基线也不自动证明新增模块有效。

## SPEED 内部测试与鲁棒性

每个条件1200张，标注框。brightness只作图像强度变换；bbox_shift为实际裁剪框向右平移，并同步更新几何条件及相对标签。selected_seed123是验证选定配置；若配置恰为full，两条结果将相同，不代表独立实验。

| 条件 | 模型 | 平移m | 旋转度 | SPEED |
|---|---|---|---|---|
| clean | baseline | 0.0230773 | 0.4490962 | 0.0099744 |
| clean | full_seed123 | 0.0228535 | 0.4491626 | 0.0099574 |
| clean | selected_seed123 | 0.0227933 | 0.4491284 | 0.0099518 |
| dark_0.5 | baseline | 0.0688410 | 0.7954752 | 0.0205803 |
| dark_0.5 | full_seed123 | 0.0681661 | 0.7964635 | 0.0205325 |
| dark_0.5 | selected_seed123 | 0.0682449 | 0.7964204 | 0.0205387 |
| bright_1.5 | baseline | 0.0746943 | 0.8620647 | 0.0213830 |
| bright_1.5 | full_seed123 | 0.0752245 | 0.8636927 | 0.0214651 |
| bright_1.5 | selected_seed123 | 0.0750895 | 0.8638098 | 0.0214551 |
| bbox_shift_10pct | baseline | 0.0297995 | 0.5497244 | 0.0122797 |
| bbox_shift_10pct | full_seed123 | 0.0296336 | 0.5503644 | 0.0122768 |
| bbox_shift_10pct | selected_seed123 | 0.0296607 | 0.5504519 | 0.0122801 |
| bbox_shift_20pct | baseline | 0.1592622 | 2.4183313 | 0.0571190 |
| bbox_shift_20pct | full_seed123 | 0.1586380 | 2.4242913 | 0.0571657 |
| bbox_shift_20pct | selected_seed123 | 0.1587931 | 2.4248040 | 0.0571865 |

逐图配对bootstrap使用3000次重采样，差值定义为新模型减基线，负值较好；区间只描述固定模型/固定样本分布的不确定性，不包含训练种子和历史测试集选择偏差。

| 条件 | 模型 | SPEED差值 | 95%下界 | 95%上界 |
|---|---|---|---|---|
| clean | full_seed123 | -0.0000170 | -0.0000351 | 0.0000013 |
| clean | selected_seed123 | -0.0000226 | -0.0000412 | -0.0000034 |
| dark_0.5 | full_seed123 | -0.0000479 | -0.0000655 | -0.0000299 |
| dark_0.5 | selected_seed123 | -0.0000416 | -0.0000599 | -0.0000237 |
| bright_1.5 | full_seed123 | 0.0000821 | 0.0000633 | 0.0001003 |
| bright_1.5 | selected_seed123 | 0.0000721 | 0.0000535 | 0.0000904 |
| bbox_shift_10pct | full_seed123 | -0.0000029 | -0.0000202 | 0.0000150 |
| bbox_shift_10pct | selected_seed123 | 0.0000004 | -0.0000174 | 0.0000195 |
| bbox_shift_20pct | full_seed123 | 0.0000468 | 0.0000305 | 0.0000628 |
| bbox_shift_20pct | selected_seed123 | 0.0000675 | 0.0000508 | 0.0000838 |

## SPEED→SPEED+ 跨域（没有域适应训练）

使用SPEED训练的模型直接评估SPEED+全部9531张真实图像，使用该数据集相机内参。报告与SPEED相同的未阈值化相对平移+角误差，用于本实验内比较，不声称等同官方SPEED+带精度阈值评分。没有使用这些真实域标签选择权重。

| 模型 | 域 | 样本数 | 平移m | 旋转度 | 未阈值化SPEED |
|---|---|---|---|---|---|
| baseline | all | 9531 | 1.0980701 | 65.3605252 | 1.3227837 |
| baseline | lightbox | 6740 | 1.0818725 | 57.6145629 | 1.1835074 |
| baseline | sunlamp | 2791 | 1.1371859 | 84.0662885 | 1.6591225 |
| full_seed123 | all | 9531 | 1.0976229 | 65.3602347 | 1.3227024 |
| full_seed123 | lightbox | 6740 | 1.0813417 | 57.6141005 | 1.1834092 |
| full_seed123 | sunlamp | 2791 | 1.1369405 | 84.0664136 | 1.6590823 |
| selected_seed123 | all | 9531 | 1.0976597 | 65.3604164 | 1.3227137 |
| selected_seed123 | lightbox | 6740 | 1.0813918 | 57.6143504 | 1.1834241 |
| selected_seed123 | sunlamp | 2791 | 1.1369452 | 84.0664301 | 1.6590849 |

## NPU网络延迟

输入已在设备上，FP32，batch1，20次预热，100次逐次同步计时；不含检测器、读图、裁剪、坐标后处理。共用机器的单次计时可能受调度影响。

| 模型 | 中位ms | p95 ms |
|---|---|---|
| baseline | 3.2662502 | 3.2796551 |
| full_seed123 | 4.4481098 | 4.4772282 |
| selected_seed123 | 4.4490825 | 4.4769682 |

## 文献对应关系与复现边界

见 [实验计划](EXPERIMENT_PLAN.md) 的逐篇页码对应，以及 [来源清单](sources.json)。FastPose-ViT 的权重衰减、框质量与域差异实验启发本轮对照；GDRNPP启发分离中心/深度监督，未复现其CAD点匹配或密集几何；UDA论文和光照论文启发独立跨域/强度压力测试，未实现UDA或物理光照仿真。tesi.pdf的讨论用于同时检查计算代价。

本轮各组仅8轮、冻结骨干、无增强，不能替代论文完整收敛训练。基线权重训练覆盖范围未知，可能与内部留出集重叠；测试集此前已观察，因此所有结果属于探索性证据。不能据此宣称论文级创新或对官方盲测的提升。

## 文件

服务器根目录：`/data/FastPose-ViT/experiments/literature_suite_20261107/run02`。
完整选定权重：`selected_model.pth`；15组最佳增量权重：`配置_seed编号/best_decoder.pt`（依赖原baseline，哈希在文件中）。训练日志为同级 `run02.log`；每组history.json记录逐轮结果。保存了脚本快照与固定protocol。
本地 [验证汇总CSV](results/validation_summary.csv)、[全部运行CSV](results/validation_runs.csv)、[压力测试CSV](results/stress_results.csv)、[跨域CSV](results/cross_domain.csv)、[延迟CSV](results/latency.csv)。

## 权重衰减数值检查

LR=5e-5、WD=1e-4 时，FP32 的衰减乘数 1−LR×WD 舍入为1.0。三个种子的 full 与 no_weight_decay 最佳解码器参数逐元素完全相同。因此这里不能推断“权重衰减无用”，只能说这档设置没有产生有效衰减。WD=0.01 的乘数为0.999999523，已补充三种子验证对照；补充实验不修改已锁定的测试模型。

| 补充配置 | 平均验证SPEED | 种子标准差 | 相对基线变化% |
|---|---|---|---|
| weight_decay_0p01 | 0.0109505 | 0.0000009 | -0.4764112 |

## 权重恢复与复现

重新运行主实验：`PYTHONPATH=. FASTPOSE_DEVICE=npu:0 OMP_NUM_THREADS=4 .venv-npu/bin/python -u scripts/run_literature_suite.py --output_dir experiments/literature_suite_new --epochs 8`（必须新建输出目录）。补充对照入口为 `scripts/run_weight_decay_control.py`，参数相同。

恢复任意增量权重：`PYTHONPATH=. .venv-npu/bin/python scripts/load_literature_checkpoint.py experiments/literature_suite_20261107/run02/full_seed123/best_decoder.pt --output experiments/full_seed123_export.pth`。脚本先验证原基线SHA256，拒绝覆盖已有输出。完整权重可通过 `FastPoseViT.from_checkpoint` 加载，再用 `forward_pose` 传入图像、框与相机内参。

## 保存权重与标准入口复核

已验证15次主实验的最优轮次与验证记录一致，去几何组的几何编码器权重保持全零，选定模型的基线骨干和原回归头逐元素不变。增量权重恢复结果与完整权重逐元素相同。

使用标准 src.evaluate 入口、NPU FP32、batch8 再测全部1200张：baseline SPEED=0.0099751437，selected=0.0099523757，约改善0.228%。与缓存评估的末位差异来自批次/FP32数值路径。另将保存的四元数在CPU float64归一化后重算指标：baseline=0.0100007201，selected=0.0099796522，仍有约0.21%的微小改善，旋转本身没有提升。该审计只提高指标计算精度，并非float64网络推理。详细数值与权重SHA256见 results/standard_evaluation_check.json。
