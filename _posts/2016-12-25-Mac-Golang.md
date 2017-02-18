---
layout:     post
title:      "Mac下Golang环境搭建"
subtitle:   "Mac下Golang环境搭建"
date:       2016-12-25
author:     "Elton"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Golang
---

# Mac下Golang环境搭建
## 下载安装Homebrew

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

## 安装Golang

```Shell
$ brew install go
```

## 设置环境变量

```Shell
$ vim ~/.profile

export GOOROOT=/usr/local/opt/go/libexec/
export GOPATH=~/study/Go
export PATH=$PATH:$GOROOT/bin:$GOPATH/bin
```

## 安装GoCode
为了支持后面Sublime的Go语言自动补全

```
$ go get github.com/nsf/gocode
$ go install github.com/nsf/gocode
```

## 安装Sublime Text3的GoSublime插件

⌘ + ⇧ + P， 调用```pakcage control```功能， 输入 ```pci```， 然后在输入```GoSublime```来安装GoSublime插件。

![GoSUblime](http://www.vckai.com/static/up/image/20150505/1432353719.png)

### 测试Hello World

```Go
package main

import "fmt"

func main() {
    fmt.Printf("hello, world\n")
}
```

![自动补全](http://www.vckai.com/static/up/image/20150505/1432354894.png)
