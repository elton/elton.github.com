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

>这里数学开始变得更复杂。**不要气馁**！我建议让笔和纸一起跟进 - 它会帮助你理解。

首先，让我们用$\frac {\partial y_{pred}}{\partial w_1}$重写偏导数：

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y_{pred}} * \frac{\partial y_{pred}}{\partial w_1}
$$

可以这样写是因为[链式求导法则](https://zh.wikipedia.org/wiki/%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99)。

我们可以计算$\frac {\partial L} {\partial y_{pred}}$，因为我们在上面已经计算了$ L =（1 - y_{pred})^2 $：

$$
\frac{\partial L}{\partial y_{pred}} = \frac{\partial (1-y_{pred})^2}{\partial y_{pred}} = -2(1-y_{pred})
$$

接下来我们要想办法获得$y_{pred}$和$w_1$的关系，我们已经知道神经元$h_1$、$h_2$和$o_1$的数学运算规则：

$$
y_{pred} = o_1 = f(w_5h_1+w_6h_2+b_3)
$$

因为$w_1$只影响神经元$h_1$，所以我们再次运用链式求导法则：

$$
\frac{\partial y_{perd}}{\partial w_1} = \frac{\partial y_{perd}}{\partial h_1}*\frac{\partial h_1}{\partial w_1}\\
\frac{\partial y_{perd}}{\partial h_1} = w_5*f’(w_5h_1+w_6h_2+b_3)
$$

用同样的方法计算$\frac{\partial h_1}{\partial w_1}$:

$$
h_1 = f(w_1x_1+w_2x_2+b_1)\\
\frac{\partial h_1}{\partial w_1} = x_1 * f'(w_1x_1+w_2x_2+b_1)
$$

$x_1$这是重量，和$x_2$是高度。这是我们第二次看到$f'(x)$（sigmoid函数的导数）现在！让我们推导出来：

$$
f(x) = \frac{1}{1+e^{-x}}\\
f'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = f(x)*(1-f(x))
$$

我们后面使用$f'(x)$这个漂亮的形式

我们完成了！我们已经设法将$\frac {\partial L}{\partial w_1}$分解成几个部分：

$$
\frac {\partial L}{\partial w_1} = \frac{\partial L}{\partial y_{pred}} * \frac{\partial y_{pred}}{\partial h_1} * \frac{\partial h_1}{\partial w_1}
$$

这种通过向后计算偏导数的系统称为**反向传播**（backpropagation）或“反向传播”。

上面的数学符号太多，下面我们带入实际数值来计算一下。$h_1$、$h_2$和$o_1$

### 例子：计算偏导数

我们将继续假设只有Alice在我们的数据集中：

Name | Weight(-135 lb) | Height(-66 in) | Gender
:--:|:--:|:--:|:--:
Alice|-2|-1|1

让我们将所有权重初始化为$1$，将所有偏差初始化为$0$.如果我们通过神经网络进行前馈传递，我们得到：

$$
\begin{aligned}
h_1 &= f(w_1x_1+w_2x_2+b_1)\\
    &= f(-2+-1+0)\\
    &= 0.0474
\end{aligned}\\
h_2 = f(w_3x_1+w_4x_2+b_2)= 0.0474\\
\begin{aligned}
o_1 &= f(w_5h_1+w_6h_2+b_3)\\
    &= f(0.0474+0.0474+0)\\
    &= 0.524
\end{aligned}
$$

神经网络的输出$y=0.524$，没有显示出强烈的是男（1）是女（0）的证据。现在的预测效果还很不好。现在计算$\frac{\partial L}{\partial w_1}$:

$$
\frac{\partial L}{\partial w_1} =  \frac{\partial L}{\partial y_{pred}} * \frac{\partial y_{pred}}{\partial h_1} * \frac{\partial h_1}{\partial w_1}\\
\begin{aligned}
\frac{\partial L}{\partial y_{pred}} &= -2(1-y_{pred})\\
                                     &= -2(1-0.524)\\
                                     &= -0.952
\end{aligned}\\
\begin{aligned}
\frac{\partial y_{pred}}{\partial h_1} &=  w_5*f’(w_5h_1+w_6h_2+b_3)\\
                                     &= 1* f'(0.0474+0.0474+0)\\
                                     &= f(0.0948)*(1-f(0.0948))\\
                                     &= 0.249
\end{aligned}\\
\begin{aligned}
\frac{\partial h_1}{\partial w_1} &= x_1 * f'(w_1x_1+w_2x_2+b_1)\\
                                  &= -2* f'(-2+-1+0)\\
                                  &= -2* f(-3)*(1-f(-3))\\
                                  &= -0.0904
\end{aligned}\\
\begin{aligned}
\frac{\partial L}{\partial w_1} &= -0.952 * 0.249 * -0.0904\\
                                  &= 0.0214
\end{aligned}
$$

>提醒：我们导出$f'(x)= f(x)* (1 - f(x))$我们之前的Sigmoid激活函数。

这告诉我们如果我们要增加$w_1$，损失函数$L$会增加非常小部分。

### 训练：随机梯度下降（Stochastic Gradient Descent）

我们现在拥有训练神经网络所需的所有工具！我们将使用一种称为随机梯度下降（SGD）的优化算法，该算法告诉我们如何改变我们的权重和偏差以最小化损失。它基本上就是这个更新等式：

$$
w_1 \gets w_1 - \eta \frac{\partial L}{\partial w_1}
$$

$\eta$是一个常数，称为学习率（learning rate），它决定了我们训练网络速率的快慢。将$w_1$减去$\eta \frac{\partial L}{\partial w_1}$，就等到了新的权重$w_1$。

* 如果 $\frac{\partial L}{\partial w_1}$为正，$w_1$就会减少，这会使$L$减少。
* 如果 $\frac{\partial L}{\partial w_1}$为负，$w_1$就会增加，这会使$L$增加。

如果我们用这种方法去逐步改变网络的权重w和偏置b，损失函数会缓慢地降低，从而改进我们的神经网络。

训练整体流程如下：

1. 从数据集中选择一个样本，这就是随机梯度下降的原因 - 我们一次只对一个样本进行操作。；
2. 计算损失函数对所有权重和偏置的偏导数，如$\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}$等；
3. 使用更新公式更新每个权重和偏置；
4. 回到第1步。

### 编码：完整的神经网络
现在终于实现了一个完整的神经网络：

Name | Weight(-135 lb) | Height(-66 in) | Gender
:--:|:--:|:--:|:--:
Alice|-2|-1|1
Bob|25|6|0
Charlie|17|4|0
Diana|-15|-6|1

![](https://victorzhou.com/media/neural-network-post/network3.svg)

```python
import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
```

完整的代码在[github](https://github.com/vzhou842/neural-network-from-scratch)

随着网络的学习，我们的损失稳步下降：

![](https://victorzhou.com/media/neural-network-post/loss.png)

我们现在可以使用网络来预测性别：

```python
# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
```

### 更多
这篇教程只是万里长征第一步，后面还有很多知识需要学习：

1、用更大更好的机器学习库搭建神经网络，如Tensorflow、Keras、PyTorch

2、在浏览器中的直观理解神经网络：https://playground.tensorflow.org/

3、学习sigmoid以外的其他激活函数：https://keras.io/activations/

4、学习SGD以外的其他优化器：https://keras.io/optimizers/

5、学习卷积神经网络（CNN）

6、学习递归神经网络（RNN）

[翻译原文](https://victorzhou.com/blog/intro-to-neural-networks/)