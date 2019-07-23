---
layout:     post
title:      "TeX中的代码高亮环境"
subtitle:   "在Tex文档中加入代码高亮"
date:       2017-12-07
author:     "Elton"
header-img: "img/blog-bg-kubernets.jpg"
header-mask:  0.3
catalog: true
multilingual: false
tags:
    - LaTex
    - code highting
---

在$\LaTeX$之中，我们可以直接使用\texttt{blabla}调出打字机字体。就现在的应用来讲，打字机字体用在代码类文本的排版比较多。在这背后，我们实际上调用的可能是cmtt12/cmtt10/cmtt9/cmtt8/cmsltt10等字体。如果使用的是XeLaTeX或者LuaLaTeX，那么我们可能是Latin Modern之中的等效字体，也就是说字形相似，但是是另外的字体了。如果我们不太老顽固的话，实际上可以选用一些OpenType格式的打字机字体来丰富一下我们文档的观感。比如最近几年O'Reilly的书中展示代码的字体已经转为Ubuntu Mono了（非Linux环境可从[Ubuntu Font Family](http://link.zhihu.com/?target=http%3A//font.ubuntu.com/)处下载）。有时候用Consolas也还行。

当然，这都不是重点。重点是代码高亮环境的内容。这是一部分带彩色的内容，受限于现有LaTeX书的印刷环境，在现有的书中讲的都很少。 这部分内容也都很简单了，比如可以看[minted vs. texments vs. verbments](http://link.zhihu.com/?target=https%3A//tex.stackexchange.com/questions/102596/minted-vs-texments-vs-verbments/102940)。我这里只讲一些最基本的例子。

## 安装pygments

首先我们需要使用python，并安装pygments：

```bash
pip install pygments
```

## 实例

之后我们准备一个简单的例子，比如（demo.tex）:

```tex
\documentclass[a4paper]{article}
\usepackage{minted}
\usepackage{xcolor}
\definecolor{bg}{rgb}{0.95,0.95,0.95}
\usepackage[margin=2.5cm]{geometry}
\begin{document}

\begin{minted}[bgcolor=bg]{rust}
fn foo(v1: Vec<i32>, v2: Vec<i32>) -> (Vec<i32>, Vec<i32>, i32) {
    // Do stuff with `v1` and `v2`.
    // Hand back ownership, and the result of our function.
    (v1, v2, 42)
}

let v1 = vec![1, 2, 3];
let v2 = vec![1, 2, 3];

let (v1, v2, answer) = foo(v1, v2);
\end{minted}

\begin{minted}[bgcolor=bg]{go}
import "math"

type Shape interface {
    Area() float64
}

type Square struct { // Note: no "implements" declaration
    side float64
}

func (sq Square) Area() float64 { return sq.side * sq.side }

type Circle struct { // No "implements" declaration here either
    radius float64
}

func (c Circle) Area() float64 { return math.Pi * math.Pow(c.radius, 2) }
\end{minted}

\end{document}
```

## 编译

然后，注意，一定要带```-shell-escape```参数：

```bash
xelatex -shell-escape demo.tex
```

或者

```bash
lualatex -shell-escape demo.tex
```

然后就出现了结果：

![](https://ws1.sinaimg.cn/large/6351ee06gy1fm8hzvq1a5j20tc0match.jpg)

## 文档
更多内容，请参阅文档：

```bash
texdoc minted
```
