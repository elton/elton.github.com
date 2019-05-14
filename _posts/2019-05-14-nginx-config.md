---
layout:     post
title:      "NGiИX在线配置"
subtitle:   "在线极速生成\"神器\""
date:       2019-05-14
author:     "Elton"
header-img: "img/blog-bg-kubernets.jpg"
header-mask:  0.3
catalog: true
multilingual: false
tags:
    - Nginx
---
# NGiИX在线配置

![NGiИX在线配置 -Elton's Blog](http://p3.pstatp.com/large/pgc-image/c5004435a264479a842f359bd6422598)

NGINX 是一款轻量级的 Web 服务器，最强大的功能之一是能够有效地提供HTML和媒体文件等静态内容。NGINX 使用异步事件驱动模型，在负载下提供可预测的性能。

是当下最受欢迎的高性能的 Web 服务器！

![NGiИX在线配置 -Elton's Blog](http://p9.pstatp.com/large/pgc-image/f404e01ece9c47c19ed0b6cbe127c8ca)

但是作为一个Nginx的使用者，你更能深有体会！你收藏，浏览，学习过很多篇 Nginx 的配置文章，但总是觉得不是很标准，在实际的使用中总是会出一些问题，咱也不敢问..就是需要自己找一些范本手动修改。

尤其是从 Apache 到 Nginx，在把文档转换为 Nginx 使用版本时，需要花费大量的时间在文档编写和文档检视上，就算这样，还是难免会出错！

为了提升广大程序员对于 Nginx 的使用幸（Sha）福（Gua）感（Shi），今天跟大家推荐一个神器！可在线极速生成Nginx配置文件的网站 ​​​​------ nginxconfig.io

![NGiИX在线配置 -Elton's Blog](http://p1.pstatp.com/large/pgc-image/ba75ce6fefce4f5596213be5d794c939)

网址：https://nginxconfig.io/

NGINX Config 支持 HTTP、HTTPS、PHP、Python、Node.js、WordPress、Drupal、缓存、逆向代理、日志等各种配置选项。在线生成 Web 服务器 Nginx 配置文件。你需要做的只需要2步：1）打开网站 2）填写相应的需求，系统就会自动生成特定的配置文件。虽然界面是英文的，但是功能的页面做的非常直观，生成的Nginx格式规范。

![NGiИX在线配置 -Elton's Blog](http://p1.pstatp.com/large/pgc-image/347ee730dc844242afeab88959ba150a)

## 实践案例

说的是轻巧，是骡子是马，咱们试试再定！

实践需求：配置域名 test.com 实现 *. test.com 自动跳转到 test.com 以及强制HTTPS（自动跳转到HTTPS）

1）填写相应的需求

![NGiИX配置在线极速生成"神器"，拒绝为Nginx配置而烦心！](http://p1.pstatp.com/large/pgc-image/70c9b8f54ebc44bab6d23d347dc784f9)

![NGiИX配置在线极速生成"神器"，拒绝为Nginx配置而烦心！](http://p9.pstatp.com/large/pgc-image/4b3d43a555824a008ce1bd4be1bbb454)

2）配置文件

![NGiИX配置在线极速生成"神器"，拒绝为Nginx配置而烦心！](http://p1.pstatp.com/large/pgc-image/76e34f2698274c8b8bb13ba497088a4d)

非常简单！非常规范！你还在为Nginx配置而烦心吗？