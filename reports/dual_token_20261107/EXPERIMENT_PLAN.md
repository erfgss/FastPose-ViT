# 文献驱动实验计划

实验对象是当前 FastPose-ViT 残差双 Token v2；不是从头复现各篇论文。所有原始权重、数据与上一轮结果保留。

## 文献依据与本次可验证问题

| 本地文献 | 阅读位置（PDF 页） | 本次采用的实验思路 | 不能声称的内容 |
|---|---|---|---|
| FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation | 5、7–8，Tables 5–9 | 预训练路径保留；权重衰减开关；框扰动与亮度敏感性检查 | 8 轮冻结分支训练不等于论文 310 轮 SPEED 训练；亮度测试不是增强训练复现 |
| GDRNPP: A Geometry-guided and Fully Learning-based Object Pose Estimator（2102.12145.pdf 实际内容为扩展版） | 4–5，Eq.5–6；9 的消融讨论 | 几何输入消融；中心 xy、深度 z 分离 L1 监督 | 本次没有 CAD 点匹配损失、密集对应或 Patch-PnP；不是 GDRNPP 复现 |
| Spacecraft Pose Estimation Based on Unsupervised Domain Adaptation and on a 3D-Guided Loss Combination | 4–6 | 将跨域评估独立列出，检查几何相关结构在真实光照域是否仍有收益 | 没有实现 PnP、深度头、伪标签或 UDA；不能称为域适应方法 |
| Machine Learning Spacecraft Pose Estimation with Realistic Illumination | 16、64 | 分别报告旋转和平移；检查暗化/亮化敏感性 | 强度缩放不是物理太阳角/BRDF 仿真，不能复现其太阳角结论 |
| AI-based Visual Pose Estimation for Space Applications（tesi.pdf） | 63–64 | 同时记录误差与 NPU 网络延迟，避免只报告精度 | 本次无关键点定位/配准/导航滤波器 |

## 固定训练协议

5 配置 × 3 随机种子（123、456、789）× 8 epoch，共 15 次运行。每次使用相同 9600 训练样本、1200 验证样本、FP32 冻结骨干特征、batch64、AdamW、LR5e-5 余弦到5e-6、梯度裁剪1。除明确开关外，训练配置相同。共同模块使用匹配初始化，样本排列使用对应种子。

- full：完整残差双 Token，几何条件、门控开启；解码后相对平移 + 归一化旋转弦距离。
- no_geometry：几何编码器所有参数置零并冻结，消除样本相关框/相机条件；视觉主干仍来自框裁剪，因此不是“完全无几何信息”。
- no_gates：关闭跨任务门控交互；两条查询和注意力保留。参数量随之变化，不能将差异完全解释为门控函数本身。
- disentangled：只替换平移损失为 crop-space xy 的 L1 和 z 的 L1，旋转损失保持一致。
- no_weight_decay：只将解码器 AdamW 的 weight_decay 从1e-4改为0。

每次在 epoch0 原始基线和8轮权重中按验证 SPEED 选最优，报告每种配置3种子的均值与样本标准差。按三种子平均验证分数选配置；最终展示模型种子预先固定123，不按测试集或最优种子选择。保存每次解码器增量权重（含基线哈希）和最终完整权重。

## 训练后固定评估

锁定配置之后才读取测试数据。SPEED 内部测试1200张：干净、像素亮度×0.5、亮度×1.5截断、框向右平移框宽10%、框向右平移20%（边界处夹紧）。框测试同时改变裁剪、几何条件、相对标签与最终坐标还原，不仅仅修改条件向量。

比较原基线、完整残差 seed123、选定配置 seed123。另在 SPEED+ 全9531张真实测试图像上做**零适应跨域评估**，分别报告 lightbox6740、sunlamp2791，使用 SPEED+ 自身相机内参。训练仍仅使用 SPEED，不与专门训练于 SPEED+ 的论文结果进行公平性声明。

计时为 NPU FP32、batch1、输入驻留设备、20次预热、100次同步测量；报告中位数和p95。它不包含检测器/数据读取/后处理，不是端到端部署速度。

## 解释边界

这是冻结分支的小预算实验，不是完整收敛训练。SPEED 内部划分来自原有标注训练集；提供的基线权重训练覆盖范围未知，可能见过留出图像。测试集此前已被查看；本轮按验证选择只减少进一步选择偏差，不能使历史测试集恢复为从未观察的最终基准。3个种子不足以宣称稳定创新；保留负结果，并将推断与实测分开。

原始文献路径、SHA256、页码见 sources.json。实验服务器目录：/data/FastPose-ViT/experiments/literature_suite_20261107/run02。
