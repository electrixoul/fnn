# FNN - 基础神经网络模型

这是 "Foundation model of neural activity predicts response to new stimulus types" (Wang et al., 2025) 论文中描述的基础神经网络模型的实现。该模型是一个详细的小鼠视觉皮层功能模型，能够跨小鼠和刺激域进行泛化。

## 项目概述

FNN（基础神经网络）是一个预测小鼠视觉皮层中神经元对各种视觉刺激反应的模型。该模型具有以下主要特点：

1. **神经反应预测**：准确预测神经元对任意视觉刺激的反应
2. **跨域泛化**：不仅能预测对自然视频（训练域）的反应，还能预测对新型刺激类型的反应，如 Gabor 滤波器、闪烁点和噪声模式
3. **迁移学习**：可以通过少量训练数据快速适应新的小鼠，优于单独训练的模型
4. **解剖学预测**：除了神经活动外，模型还捕捉解剖学特性，可以预测细胞类型和连接性

## 模型架构

神经网络由四个主要模块组成：

1. **透视模块** (`fnn/model/perspectives.py`)：
   - 使用光线追踪技术根据眼睛位置和方向转换视觉刺激
   - 考虑小鼠对监视器刺激的视角
   - 将视网膜建模为接收光线的球面上的点

2. **调制模块** (`fnn/model/modulations.py`)：
   - 处理行为变量（运动、瞳孔大小）
   - 使用 LSTM 维持小鼠行为状态的动态表示
   - 输出影响视觉处理的调制特征图

3. **核心模块** (`fnn/model/cores.py`)：
   - 包含模型的主要容量
   - 由前馈（3D 卷积）和循环（LSTM）组件组成
   - 前馈组件使用 DenseNet 架构
   - 循环组件使用 CvtLstm（卷积视觉变换器 LSTM）

4. **读出模块** (`fnn/model/readouts.py`)：
   - 将核心输出映射到单个神经元活动
   - 对于每个神经元，使用两个参数：空间位置和特征权重
   - 这些参数作为与解剖特性相关的"功能条形码"

## 安装

可以直接从 GitHub 安装该库：

```bash
pip install git+https://github.com/cajal/fnn.git
```

或从本地源代码安装：
```bash
git clone https://github.com/cajal/fnn.git
cd fnn
pip install -e .
```

## 基本使用

```python
from fnn import microns
from numpy import full, concatenate

# 加载 MICrONS scan 8-5 的模型和神经元 ID
model, ids = microns.scan(session=8, scan_idx=5)

# 示例 3 秒视频（3 x 30 帧 @ 30 FPS，144 高，256 宽）
frames = concatenate([
    full(shape=[30, 144, 256], dtype="uint8", fill_value=0),   # 1 秒黑屏
    full(shape=[30, 144, 256], dtype="uint8", fill_value=128), # 1 秒灰屏
    full(shape=[30, 144, 256], dtype="uint8", fill_value=255), # 1 秒白屏
])

# 预测神经元对 3 秒视频的反应
response = model.predict(stimuli=frames)
```

## 示例

本库中包含两个示例脚本：

1. `test_fnn.py`：演示基本用法，加载模型并预测对简单黑-灰-白序列的神经反应
2. `test_fnn_advanced.py`：演示高级用法，包括不同刺激类型的预测和结果可视化

运行示例：

```bash
# 基本测试
python test_fnn.py

# 高级测试（生成可视化）
python test_fnn_advanced.py
```

高级测试将生成两个文件：
- `fnn_responses.png`：选定神经元对不同刺激类型的反应可视化
- `receptive_fields.png`：神经元感受野位置的空间分布可视化

## 科学意义

这个基础模型展示了人工智能技术如何推进神经科学研究。通过组合来自多个实验的数据，包括来自自然条件下的许多脑区和受试者的数据，该模型创建了小鼠视觉系统的统一表示。它捕捉了神经元和受试者之间的相似性，同时考虑到个体差异，提供了对神经编码和视觉处理的前所未有的洞察。
