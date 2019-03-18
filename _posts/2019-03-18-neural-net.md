---
layout:     post
title:      "神经网络"
subtitle:   "从原理到优化"
date:       2019-03-18
author:     "Elton"
header-img: "img/blog-bg-kubernets.jpg"
header-mask:  0.3
catalog: true
multilingual: false
tags:
    - 算法
    - 神经网络
---

## 1. 搭建基本模块——神经元

在说神经网络之前，我们讨论一下**神经元**（Neurons），它是神经网络的基本单元。神经元先获得输入，然后执行某些数学运算后，再产生一个输出。比如一个2输入神经元的例子：

![](https://victorzhou.com/media/neural-network-post/perceptron.svg)

在这个神经元中，输入总共经历了3步数学运算，先将两个输入乘以权重（weight）(红色部分)：

$$
x_1 \rightarrow x_1∗w_1\\
x_2 \rightarrow x_2 * w_2
$$

把两个结果想加，再加上一个偏置$b$(bias)(绿色部分)：

$$
(x_1 ∗ w_1)+(x_2 ∗ w_2)+b
$$

最后将它们经过激活函数（activation function）处理得到输出(黄色部分)：

$$
y=f(x_1*w_1+x_2*w_2+b)
$$

激活函数的作用是将无限制的输入转换为可预测形式的输出。一种常用的激活函数是[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)函数：
![](https://victorzhou.com/media/neural-network-post/sigmoid.png)

**sigmoid函数**的输出介于$0$和$1$，我们可以理解为它把$(-\infty, +\infty)$范围内的数压缩到 $(0,1)$以内。正值越大输出越接近$1$，负向数值越大输出越接近$0$。

### 一个例子
假设我们有一个2输入神经元，它使用sigmoid激活函数并具有以下参数：

$$
\begin{aligned}
   w&=[0,1] \\
   b&=4
\end{aligned}
$$

$w=[0,1]$是$w_1=0$、$w_2=1$的向量形式写法。给神经元一个输入$x=[2,3]$，可以用向量[点积](https://simple.wikipedia.org/wiki/Dot_product)的形式把神经元的输出计算出来：

$$
\begin{aligned}
   (w \cdot x)+b &=((w_1*x_1)+(w_2*x_2))+b \\
                 &=0*2+1*3+4 \\
                 &=7
\end{aligned}\\
y=f(w\cdot x+b) =f(7)=0.999
$$

给定输入$x = [2,3] $，神经元输出$0.999$。给与输入然后得到输出的过程称为**前馈**(feedforward)。

### 编码一个神经网络
是时候实施一个神经元了！我们将使用[NumPy](http://www.numpy.org/)，一个流行且功能强大的Python计算库来帮助我们进行数学运算：
```python
import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994
```

## 2. 搭建神经网络
神经网络就是把一堆神经元连接在一起，下面是一个神经网络的简单举例：
![](https://victorzhou.com/media/neural-network-post/network.svg)

这个网络有2个输入、一个包含2个神经元的隐藏层（$h_1$和$h_2$）、包含1个神经元的输出层$o_1$。请注意$o_1$的输入是$h_1$和$h_2$输出 - 这就是网络的原因。

>**隐藏层**是夹在输入层（第一个层）和输出层（最后一个层）之间的部分，一个神经网络可以有多个隐藏层。

### 一个例子：前馈（Feedforward）
我们假设上面的网络里所有神经元都具有相同的权重$w=[0,1]$和偏置$b=0$，激活函数都是sigmoid，设$h_1$，$h_2$，$o_1$表示它们所代表神经元的输出。
如果我们传入输入$x = [2,3]$会发生什么？
$$
\begin{aligned}
   h_1 = h_2 &= f(w\cdot x+b) \\
             &= f((0*2)+(1*3)+0)\\
             &=f(3)\\
             &=0.9526
\end{aligned}\\
\begin{aligned}
         o_1 &= f(w\cdot [h_1,h_2] + b)\\
             &= f((0*h_1)+(1*h_2)+0)\\
             &= f(0.9526)\\
             &= 0.7216
\end{aligned}
$$
输入$x = [2,3]$的神经网络，输出是$0.7216$。很简单吧。

神经网络可以具有**任意数量的层**，这些层中具有**任意数量的神经元**。基本思想保持不变：给神经网络提供输入(input)通，然后从神经网络里面得到输出(output)。为简单起见，我们将继续使用上图所示的网络来完成本文的其余部分。

### 代码实现：前馈（Feedforward）
让我们为神经网络实现前馈。这是网络的图像再次供参考：
![](https://victorzhou.com/media/neural-network-post/network.svg)

```python
import numpy as np

# ... code from previous section here

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    # The Neuron class here is from the previous section
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.7216325609518421
```

## 3. 训练一个神经网络（第一部分）
现在我们已经学会了如何搭建神经网络，现在我们来学习如何训练它，其实这就是一个优化的过程。
假设我们有以下测量值，包含4个人的身高、体重和性别：
Name | Weight(lb) | Height(in) | Gender
:--:|:--:|:--:|:--:
Alice|133|65|F
Bob|160|72|M
Charlie|152|70|M
Diana|120|60|F
现在我们的目标是训练一个网络，根据体重和身高来推测某人的性别。
![](https://victorzhou.com/media/neural-network-post/network2.svg)
为了简便起见，我们将每个人的身高、体重减去一个固定数值，把性别男定义为1、性别女定义为0。
Name | Weight(-135 lb) | Height(-66 in) | Gender
:--:|:--:|:--:|:--:
Alice|-2|-1|1
Bob|25|6|0
Charlie|17|4|0
Diana|-15|-6|1

>我随意选择了移位量（135和66）以使数字看起来不错。通常情况下，你会是平均值。

### 损失（Loss）
在我们训练网络之前，我们首先需要一种方法来量化它的“好”程度，以便它可以尝试“更好”。这就是**损失**(Loss)。
比如用均方误差（mean squared error,MSE）来定义损失：

$$
MSE=\frac{1}{n}\sum_{i=1}^n (y_{ture}-y_{pred})^2
$$

* $n$是样本数，即4个人（Alice，Bob，Charlie，Diana）
* $y$表示预测的变量，即性别(Gender)。
* $y_{ture}$ 是变量的真实值（“正确答案”）。例如，Alice的 $y_{true}$ 是1（女）。
* $y_{pred}$是变量的预测值，也是我们神经网络的输出值。

$(y_{true}-y_{pred})^2$,被称为**平方误差，简称方差(squared error.)**。顾名思义，均方误差就是所有数据方差的平均值，我们不妨就把它定义为损失函数。预测结果越好，损失就越低，训练神经网络就是将损失最小化。

好的预测=最低的损失
**训练一个网络 = 尝试最小化损失**

### 损失计算的例子
如果上面网络的输出一直是0，也就是预测所有人都是男性，那么损失是：

Name | $y_{true}$ | $y_{pred}$ | $(y_{true}-y_{pred})^2$
:--:|:--:|:--:|:--:
Alice|1|0|1
Bob|0|0|0
Charlie|0|0|0
Diana|1|0|1

$$
MSE=\frac{1}{4}(1+0+0+1) = 0.5
$$

### 编码 MSE
```python
import numpy as np

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5
```
如果您不理解此代码的工作原理，请阅读有关数组操作的[NumPy快速入门](https://docs.scipy.org/doc/numpy/user/quickstart.html#basic-operations)。

## 4.训练一个神经网络（第二部分）
我们现在有一个明确的目标：**尽量减少神经网络的损失**。我们知道我们可以改变网络的权重和偏差以影响其预测，但我们如何以减少损失呢？
>本节使用了一些多变量微积分。如果您对微积分不感兴趣，请跳过数学部分。

为了简单起见，我们把数据集缩减到只包含Alice一个人的数据。

Name | Weight(-135 lb) | Height(-66 in) | Gender
:--:|:--:|:--:|:--:
Alice|-2|-1|1

于是损失函数就剩下Alice一个人的方差：

$$
\begin{aligned}
         MSE &= \frac{1}{1}\sum_{i=1}^1(y_{true}-y_{pred})^2 \\
             &= (y_{true}-y_{pred})^2\\
             &= (1-y_{pred})^2
\end{aligned}
$$

考虑损失的另一种方式是权重和偏差。让我们在网络中标出每个权重和偏见：
![](https://victorzhou.com/media/neural-network-post/network3.svg)

然后，我们可以将损失写为多变量函数：

$$
L(w_1,w_2,w_3,w_4,w_5,w_6,b_1,b_2,b_3)
$$

想象一下，我们想要调整$w_1$ 。如果我们改变了$w_1$，L损失函数的值会如何改变 ？这是[偏导数](https://zh.wikipedia.org/wiki/%E5%81%8F%E5%AF%BC%E6%95%B0)$\frac {\partial L}{\partial w_1}$可以回答的问题。我们如何计算呢？