---
tags:
- 永久笔记
- ai
- 网络安全
- llm
layout: post
date: '2025-12-19 23:51:10'
title: Universal and Transferable Adversarial Attacks on Aligned Language Models
author: jkofbr
header-img: img/header_img.jpg
catalog: true
---

# 正文
[论文正文](https://arxiv.org/pdf/2307.15043)
## 总览
![1000](/img/file-20251210184041465.png)
## 损失函数的建立
我们可以将这一目标写成对抗性攻击的正式损失函数。我们将大语言模型视为从某个token序列$x_{1: n}$的映射，其中$x_{i} \in{1, ..., V}$（V表示词汇量，即令牌的数量）映射到下一个令牌的分布。具体来说，对于任何$x_{n+1} \in{1, ..., V}$，我们使用符号
$$
p\left(x_{n+1} | x_{1: n}\right), (1)
$$
来表示给定前序令牌的情况下，下一个token是$x_{n+1}$的概率。
tokens $x_{1: n}$ 。稍微滥用一下符号，记$p(x_{n+1: n+H} | x_{1: n})$表示在序列$x_{n+1: n+H}$中，给定该位置之前的所有token的情况下，生成每个单个token的概率，即 $$p\left(x_{n+1: n+H} | x_{1: n}\right)=\prod_{i=1}^{H} p\left(x_{n+i} | x_{1: n+i-1}\right)$$
### 例子：当目标序列有 2 个 token（H=2）时
- 目标序列是$x_n+1​,x_n+2$​，此时需要计算 “同时生成这两个 token” 的联合概率$p(x_n+1​,x_n+2​∣x_{1:n}​)。$
- 根据**概率链式法则**（联合概率 = 先验概率 × 条件概率）：$p(AB∣C)=p(A∣C)\times{p(B∣AC)}$
- ![Drawing_2025-12-11_01.04.19.excalidraw](/img/Drawing_2025-12-11_01.04.19.excalidraw.md)
- 对应到 token 序列中：$p(x_{n+1:n+2}​∣x_{1:n​})=p(x_{n+1}​∣x_{1:n}​)×p(x_{n+2}​∣x_{1:n}​,x_{n+1}​)$
### 建立损失函数
$$
\mathcal{L}(x_{1:n​})=−\log{p(x_{n+1:n+H}^⋆​∣x_{1:n}}​)
$$
- 目标序列 $x_{n+1:n+H}^⋆​$：代表我们希望 LLM 生成的 “肯定性开头”（如 “Sure, here is how to build a bomb.”），是攻击成功的关键 “触发信号”。
- 条件概率 $p(x_{n+1:n+H}^⋆​∣x{1:n})$：结合前文公式，它表示 “在输入序列 x1:n​（含用户有害查询 + 对抗性后缀）的条件下，LLM 生成目标序列 $x_{n+1:n+H}^⋆​$的联合概率”。
#### 选择该损失函数(-log)的理由
![Drawing_2025-12-11_01.29.33.excalidraw](/img/Drawing_2025-12-11_01.29.33.excalidraw.md)
- **单调性一致**：对数函数是单调递增函数，因此 logp(⋅) 与 p(⋅) 单调性一致；加负号后，“最大化 p(⋅)” 等价于 “最小化 −logp(⋅)”，符合优化算法（如梯度下降）“最小化损失” 的常规逻辑。
- **数学易处理**：概率 p(⋅) 是多个 token 概率的乘积（见前文公式 $$p\left(x_{n+1: n+H} | x_{1: n}\right)=\prod_{i=1}^{H} p\left(x_{n+i} | x_{1: n+i-1}\right)$$，乘积易导致数值下溢（多个 0~1 的数相乘，结果会趋近于 0）；而对数能将 “乘积” 转化为 “求和”（${\log{\prod_{i=1}^{H}{a_i}}=\sum_{i=1}^{H}\log{a_i}}$​），既避免下溢，又简化梯度计算。
#### 使用损失函数进行优化
因此，优化对抗性后缀的任务可以写成优化问题：
$$\underset{x_{\mathcal{I}} \in\{1, ..., V\}^{|\mathcal{I}|}}{minimize} \mathcal{L}\left(x_{1: n}\right) (4)$$其中 $\mathcal{I} \subset\{1, ..., n\}$ 表示LLM输入中对抗性后缀标记的索引

-   
$x_{\mathcal{I}} \in\{1, ..., V\}^{|\mathcal{I}|}$表示 “对抗性后缀的每个 token，都必须从词汇表的V个候选中选择”
- 在 “仅修改对抗性后缀、且后缀 token 必须来自 LLM 词汇表” 的约束下，找到最优的后缀 token 序列，让输入序列诱导 LLM 生成 “目标肯定性开头” 的损失最小（即概率最大），最终实现对抗性攻击。
## 算法的实现
### 算法1
![](/img/file-20251211194347800.png)
#### 重点部分解释
- ${-\nabla_{e_{x^i}}{\mathcal{L}(x_{1:n})}}$
	- $e_{x^i}$: [[one-hot]] 向量
	- 表示如果将当前 token 改为词表中其他候选 token 时 loss 的变化趋势。

##### 📌 如何具体计算梯度 ∇ₑₓᵢ L(x₁:ₙ)

1. **正向传播计算 loss**
    
    - 给定 $prompt (x_{1:n})$，模型先把所有 token 通过 embedding 层转成向量，再经过 transformer 等结构计算模型输出。
    - 损失 (L) 是模型输出与某个目标行为（例如生成某种输出）的函数。
2. **反向传播到 one-hot 表示**
    - 通过标准反向传播，我们可以把损失对每个 embedding 向量求偏导:$$\frac{\partial L}{\partial \text{embedding}(x_i)}.$$
    - 因为 embedding 是 one-hot 向量 $e_{x_i}$ 与 embedding 矩阵 (E) 的乘积，  
    - $$
        \text{embedding}(x_i) = E^\top e_{x_i},  $$
    - 所以损失对 one-hot 向量的梯度可以通过链式法则得到：  $$\nabla_{e_{x_i}} L = E \cdot \frac{\partial L}{\partial \text{embedding}(x_i)}.$$
        换句话说：
        - 你把 Loss 对当前 token 的 embedding 的梯度向量乘回 embedding 矩阵，就可以得到 Loss 对 **one-hot 代表的每个词表位置** 的梯度值。
    - 这个结果是一个长度等于词表大小 (|V|) 的向量：  $$\nabla_{e_{x_i}}L(x_{1:n}) \in \mathbb{R}^{|V|},$$
    - 它告诉你“如果把当前 token 替换成词表中的其他 token，会如何改变 loss”。

