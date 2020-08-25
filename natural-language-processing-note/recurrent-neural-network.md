---
description: A very drafty summary.
---

# Recurrent Neural Network

## General Recurrent Neural Network

可以参考fast.ai note中lesson7的最后。这个[link](https://www.youtube.com/watch?v=LHXXI4-IEns)也很棒。

RNN是个序列模型，序列模型顾名思义就是一个一个去process句子里面的单词，每个单词embedding之后变成vector进入到模型当中。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2tx1rw2qg30qe06yh7f.gif)

他们相互计算的方式，就是一个for loop，上一个loop承载着以前的记忆，合并到新进来的单词里面。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2typi3e9g30qe0agjwr.gif)

而Tanh function是一个hiddent state传输数据需要经过的激活层，才能传递到下一个hiddent state中。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2tyasysqg30qe06y0ty.gif)

这个Tanh是干嘛的呢，Tanh的作用就是给输出的matrix一个bound，

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2u3ayh04g30qe0agady.gif)

下面图中是没有tanh的，每一层都做了一个乘3的计算，发现有些数字会out weighted。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2u2cscikg30qe03in5j.gif)

而有了Tanh之后，值会被bound在-1到1之间

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2u31v31bg30qe03iqbw.gif)

## LSTM

[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

LSTM和GRU的出现主要是为了解决单纯的RNN的vanishing gradient的问题，a.k.a 金鱼记忆。 没有办法把之前的单词记忆传送到后来新的单词当中，前面层的变化越来越小，gradient也越来越小，导致了不训练。

针对上述的问题，LSTM和GRU中有一个叫Gates的东西，regulate the flow of information。它像一个火车轨道的方向器，可以控制sequence是被忘记还是被记住，有点像batch normalizaiton？ 就好比啦你看八卦的时候，不会每个单词都记住，但是keyword总是能一下抓住你的眼球。所以这个Gate就是掌握和选择relevent information的关卡。

![image-20200623121812221](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2tqziq5sj31p30u07n2.jpg)

**我觉得值得注意的是，上图中是有两个flow的，上面的flow是cell state可以理解是至今为止你学到的东西，他是通过了sigmoid和tanh的选择和学习的，里面包括的东西有旧的知识和新的知识的结合。而下面的hidden state就是对于现阶段来说以前的纯粹的旧的东西，是要去决定哪一部分需要忘记的。在下面说道的各种gate中，他们的流入部分也不太一样，起到的作用也不太一样。**

这个Gate呢，可以说是另一种neural network。主要是运用了sigmoid function。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2vw097gwg30qe0ag41f.gif)

### Forget gate

Forget gate流入的信息是纯粹的旧的信息。他需要去判断这个information是否被继续使用，有点像是一个二分类器，最后输出一个Probability告诉你他应该被舍弃还是应该被留下，既然说到了probability那么sigmoid function是无法避免的！图中红色的部分就是sigmoid function，

> This gate decides what information should be thrown away or kept. Information from the previous hidden state and information from the current input is passed through the sigmoid function. Values come out between 0 and 1. The closer to 0 means to forget, and the closer to 1 means to keep.

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2vxzsal4g30qe0dwafi.gif)

### Input gate

这个gate的流入信息是旧的和新的结合，所以他是去 decide what new information we’re going to store in the cell state. This has two parts.

第一个部分是让sigmoid选择我们要update什么，这个具体的sigmoid的部分就是“input gate layer” ，有点像是做出一个权重。

第二个部分则是直接不通过选择的进入tanh做计算，这个和basic RNN是一样的。

最后decide what to store就是通过两者点成而得出的。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2wn3k7v4g30qe0ci0za.gif)

### Update cell state

要update cell state得到流进下一个cell的信息，还需要结合what to forget的部分也就是forget gate的output。

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2wt0kzkmg30qe0dwwij.gif)

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2wtdoj74j31eq0fo0ub.jpg)

### Output gate

到了最后一个gate，他是为了决定下一个hidden state内容的。要和新算出来的cell state结合通过tanh算出下一个hidden state。在给下一个的hidden state里面，我包括了这个state的新东西，上个state的旧东西，以及这个state我通过权重学习到的东西。

![img](https://miro.medium.com/max/1400/1*VOXRGhOShoWWks6ouoDN3Q.gif)

## GRU

So now we know how an LSTM work, let’s briefly look at the GRU. The GRU is the newer generation of Recurrent Neural networks and is pretty similar to an LSTM. GRU’s got rid of the cell state and used the hidden state to transfer information. It also only has two gates, a reset gate and update gate.

![img](https://tva1.sinaimg.cn/large/007S8ZIlgy1gg2x2ixeswj30sy0nm0u3.jpg)

### Update gate

The update gate acts similar to the forget and input gate of an LSTM. It decides what information to throw away and what new information to add.

### Reset Gate

The reset gate is another gate is used to decide how much past information to forget.

And that’s a GRU. GRU’s has fewer tensor operations; therefore, they are a little speedier to train then LSTM’s. There isn’t a clear winner which one is better. Researchers and engineers usually try both to determine which one works better for their use case.

