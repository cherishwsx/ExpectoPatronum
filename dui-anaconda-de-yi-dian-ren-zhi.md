---
description: 这里有一条蟒蛇
---

# 对Anaconda的一点认知

昨天下午在做Intern时候入门了一下AWS，然后突发奇想想把自己电脑上的Python的各种版本给整理一下,，然后换成Anaconda的。接着就开始了我昨晚的崩溃三小时。

#### 1. 整理已有的Python版本

首先我知道我电脑里面Python的版本有很多很多，之前直接安装过Python3官方的IDE就，通过brew也安装过，Python2好像也有安装。总之就是零零散散各种版本。那我该怎么办，于是我就想办法把`\usr\local\bin`里自带的Python给整个带走了，然后再`brew clean`了一下。主要是参考了这一篇 [https://www.jianshu.com/p/98383a7688a5](https://www.jianshu.com/p/98383a7688a5) 的教学。但是我最后输入`python3`之后，还是会给我进入Python中，并且所指向Python路径是`/usr/bin`中，这个路径按道理来说是System中的路径不能随意改动的，所以后来我也没敢乱动这里。在整理完Python各种版本之后，我的Pycharm里面剩下这些：（其实到现在我也还是不明白这几个有什么区别，但是我不敢动）

![](.gitbook/assets/image.png)

这两个是外来的Python版本，其中invalid是我清理掉的，另一个是Anaconda的环境

#### 2. 安装Anaconda

讲道理装好Anaconda之后，是自动会把PATH加入进bash file的，如果遇到这个问题请自行Google。下面是每次都会查的conda命令大全：

{% embed url="https://blog.csdn.net/menc15/article/details/71477949" %}

然后你就可以开心的建立各种环境啦！因为我下载的是Anaconda3所以base环境下就直接是Python3

![](.gitbook/assets/image%20%282%29.png)

当你创建了新的环境的时候，你的新环境是没有Anaconda的自带包的，所以可以通过`conda install anaconda`到这个环境当中去获取所有包的权限

#### 3. Jupyter Notebook Kernel设置

你可以去调配你的kernel使用的是哪个编译器，为了方便我把我的之前使用自带Python3的kernel编译器改到了base下。首先查看一下你的kernel有些，并且找到所在位置的kernel.json文件去查找你编译器的位置，来根据你的需要来更改。

`jupyter kernelspec list`

参考：[https://blog.csdn.net/baoqiaoben/article/details/82912189](https://blog.csdn.net/baoqiaoben/article/details/82912189)

#### 4. Pycharm环境设置

为了使用Anaconda的自带环境，所以把Pycharm的环境设置为Anaconda的环境。一定要从路径下找这个文件，而不是直接选Anaconda environment。

![](.gitbook/assets/image%20%283%29.png)

#### 5. 这是一个漫长的过程

真心感觉到调配环境是一个冗长的过程。之后到AWS的EC2上跑Jupyter也不知道是个什么情况。Good Luck for me。

