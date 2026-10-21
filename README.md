# ML & DL 学习项目

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Scikit-learn-ML-green" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-purple" alt="NLP" />
</p>

一个以 Python 为核心、覆盖机器学习（Machine Learning）、深度学习（Deep Learning）和自然语言处理（NLP）的学习仓库。项目内容来源于课程实践、案例演示和代码实验，适合用于 AI 入门学习、课程复盘和模型实战训练。

## 项目概览

本仓库按学习方向分为三个模块：

- `ML/`：传统机器学习算法与数据分析实践
- `DL/`：PyTorch 深度学习基础、网络构造与优化
- `NLP/`：文本处理、情感分析、RNN/LSTM、FastText 与 AI 应用

同时包含：

- `README.md`：项目说明
- `runs/`：训练日志和 TensorBoard 产物
- `data/`：训练与测试数据集

## 目录结构

```text
ML&DL/
├── README.md
├── DL/
│   ├── day01/
│   ├── day02/
│   ├── day03/
│   ├── day04/
│   ├── day05/
│   ├── day06/
│   ├── day07/
│   ├── day08/
│   └── data/
├── ML/
│   ├── day01/
│   ├── day02/
│   ├── day03/
│   ├── day04/
│   └── data/
├── NLP/
│   ├── day01/
│   ├── day02/
│   ├── day03/
│   ├── day04/
│   ├── day05/
│   └── ...
├── runs/
├── .git/
├── .vscode/
└── .idea/
```

## 学习模块

### 1. 机器学习（`ML/`）

`ML/` 目录主要包含以下内容：

- 线性回归与正规方程法
- KNN 分类与回归
- KMeans 聚类
- 决策树、随机森林
- 模型评估与交叉验证
- 客户案例分析与数据挖掘

示例文件：

- `ML/day02/鸢尾花识别.ipynb`
- `ML/day02/正规方程式解决问题.ipynb`
- `ML/day03/cart-泰坦尼克号生存预测-分类问题.py`
- `ML/day04/KMeans模型.py`
- `ML/day04/随机森林实现泰坦.py`

### 2. 深度学习（`DL/`）

`DL/` 目录主要聚焦 PyTorch 深度学习基础与实战：

- 张量创建与基本运算
- 神经网络构造与结构设计
- 激活函数、损失函数与优化器
- CNN、RNN、LSTM、GRU
- 图像分类、图像分析与词嵌入

示例文件：

- `DL/day01/torch初体验.ipynb`
- `DL/day03/构造神经网络.py`
- `DL/day04/网络优化.ipynb`
- `DL/day06/图像分析案例.py`
- `DL/day07/词嵌入层.ipynb`

### 3. 自然语言处理（`NLP/`）

`NLP/` 目录主要包括：

- 文本预处理与中文分词
- 词嵌入与 Embedding
- RNN / LSTM / GRU 结构
- 酒店评论情感分析
- FastText 文本分类
- 大模型相关应用脚本

示例文件：

- `NLP/day02/fasttext_train.py`
- `NLP/day02/hotel_analys.py`
- `NLP/day03/lstm_demo.py`
- `NLP/day03/rnn_demo.py`
- `NLP/day03/hotel_pre.py`
- `NLP/day05/ai_chat.py`

## 技术栈

本项目主要使用：

- Python
- PyTorch
- scikit-learn
- NumPy
- pandas
- matplotlib
- jieba
- FastText
- Jupyter Notebook

## 运行环境建议

建议使用 Python 3.8 及以上版本，并安装以下依赖：

```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib jieba
```

如果需要运行 notebook，建议使用 VS Code + Jupyter 扩展，或在本地配置 Python 环境后直接运行。

## 学习路线建议

推荐按照以下顺序学习：

1. Python 基础与数据处理
2. 机器学习基础算法
3. 模型评估与交叉验证
4. PyTorch 张量与神经网络基础
5. CNN / RNN / LSTM / GRU
6. NLP 文本预处理与词嵌入
7. 情感分析与文本分类实战
8. 扩展到大模型与 AI 应用

## 项目特点

- 内容偏教学与实战结合
- 覆盖传统机器学习、深度学习和 NLP 三大方向
- 结合 notebook 与脚本文件，便于学习和复现
- 包含多个真实案例和训练流程
- 适合入门学习和中期强化训练

## 结语

这个仓库是一个比较完整的 AI 学习项目，聚焦于：

- 机器学习基础
- 深度学习与 PyTorch
- 自然语言处理
- 文本分类与模型训练

适合正在学习 Python、机器学习、深度学习和 NLP 的开发者使用。通过本仓库，可以逐步建立起从基础算法到实际模型训练的知识框架。
