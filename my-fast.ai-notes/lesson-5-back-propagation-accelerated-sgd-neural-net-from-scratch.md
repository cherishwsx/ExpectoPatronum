---
description: 'Detailed Notes: https://github.com/hiromis/notes/blob/master/Lesson5.md'
---

# Lesson 5: Back propagation; Accelerated SGD; Neural net from scratch

## Lesson 5

## Review

![flow](https://tva1.sinaimg.cn/large/0082zybpgy1gbwqcrxdynj321r0u01bo.jpg)

整个DNN流程中，有两类层，一类是存储参数值的layer，一类是存储激活值的layer\(element-wise function\)。对于存储参数值的Layer来说，我们用梯度下降去update这些参数:`parameters ﹣= learning rate * parameters.grad`. 我们输入激活值\(Inpur layer activation numbers are your data, otherwise it's your last step results\)和weight做matrix multiplication. 整个流程如下。

![image-20200221160149626](https://tva1.sinaimg.cn/large/0082zybpgy1gc4uq15cukj30u0107jti.jpg)

```text
graph TD
    A[Data] --> B[DNN Layer 1]
    B -->|FP Compute Loss| C[DNN Layer 2]
    C -->|BP Gradient Descent & LR to Update Parameter| B
    C -->|FP| D[...]
    D -->|BP| C
    D -->|FP| E[Last Activation Layer]
    E -->|BP| D
    E -->|BP| F[Output]
    F -->|FP| E
    F --> G[Groud Truth]
```

### Back Propagation

So that piece where we take the loss function between the actual targets and the output of the final layer \(i.e. the final activations\), we calculate the gradients with respect to all of these yellow things, and then we update those yellow things by subtracting learning rate times the gradient `parameters ﹣= weight.grad * learning rate`. That process of calculating those gradients and then subtracting like that is called **back propagation**.

### Fine Tuning

​ 对于Transfer Learning中ResNet34来打比方，the first thing to notice is the ResNet34 has a 1000 columns weight matrix since there are 1000 categories in ImageNet. 但是通常我们的label和ImageNet Pretrained model中的label是不一样的，所以我们不需要最后一个将value输出位1000 columns的那个matrix，而fast.ai会帮我们删除并且帮我们替换成了一个matrix，ReLu已经另一个matrix， 第一个matrix是由上一层输出的size决定的，fast.at帮你default好了，而第二个matrix是由你的`data.c`决定的。By Default， 这两个matrix是randomized的 and need to be trained. But other matrices, they are pretrined with ImageNet data. Recall [Zeiler and Fergus paper](https://arxiv.org/pdf/1311.2901.pdf).

​ 更形象的说，在ImageNet中可能最后CNN能捕捉到的是眼球，但是你的data中可能没有眼球，所以你不需要一个可以detect出眼球这么具体物体的matrix，所以你替换成你自己的。

#### Freeze

​ 之后通过Transfer Leanring，将前面对我们有用的matrix freeze也就是不参与任何参数更新和训练，从而达到

1. Have less training time
2. Takes up less memory since less gradient that has to store
3. Better than random initialization

![ImageNet](https://tva1.sinaimg.cn/large/0082zybpgy1gc1hqm0ylcj30xr0hi11d.jpg)

#### Unfreeze

​ 当freeze结束之后，如果效果还不错，就unfreeze前面的matrix， 让他们参与训练但是值得注意的是，这时matrix不是random Initialized。所以可能他们不需要太多的训练因为这些参数还是可以识别边边角角和一些简单的物体，而且这些东西是对我们有用的。而我们新加入的那两个matrices，他们是需要更多的训练的，这就是为什么我们需要assign different learning rate。前面不需要怎么训练的matrices就学的慢一点，因为他们很可能已经在一个比较好的min了，更大的learning rate可能会让他们被kicked out到一个不好的地方去。而后面新的matrices就学的快一点。It's called **discrimative learning rate**.

![image-20200218183321076](https://tva1.sinaimg.cn/large/0082zybpgy1gc1i8t52sqj31p00u04qp.jpg)

### discrimative learning rate

Rule of thumbs is 3e-3。

* All the layers get the same learning rate. 

```python
fit(1, 1e-3)
```

* All the layer except the last layer get learning rate 1e-3/3, and the last layer is 1e-3. 这里为什么除以3呢，是和Batch Normalization有关的。

```python
fit(1, slice(1e-3))
```

* The first layer get 1e-5, the other layer will get the learning rate that are multiplicatevelly equal between these two number. For example, if you have three layers, then the learning rate for each layer would be 1e-5, 1e-4, 1e-3.

```python
fit(1, slice(1e-5, 1e-3))
```

We don't give different learning rate to each different layer, we give different learning rate to different layer groups. Fast.ai grouped the new added layers as one layer group and divided existing layers into two groups, so for CNN you will have three layers.

### Collaborative Filtering

在协同过滤中，我们是这么写的

![image-20200219183822658](https://tva1.sinaimg.cn/large/0082zybpgy1gc2o0dapitj30jg052wfl.jpg)

是因为协同过滤只有一层layer，并不是传统的matrix mul + activation layer + matrix mul。

#### Embedding Matrix

One matrix for users and one matrix for movies. And they are randomly initialized waited to be optimized. And they are called embedding. 我觉得这一段\[[22:51](https://youtu.be/uQtTwhpv7Ew?t=1371)\]，Jeremy的解释有一点混乱，需要对这一段多听几遍。简单来说Embedding是object对于某些特征的表达性，比如一个user对于Action，Love and Adventure这个三个feature是什么样的preference？或者这三个类型又是如何表达一个电影的？我们在gradient descent中就是在optimize这两个embedding matrices，让他们更能代表这些user，这些电影，其实face recognition当中也是一样的，找出最能代表一个人脸的embedding。然后通过这两个matrices的相乘，来判断一个user对于一个电影的喜爱程度。在这里呢，Jeremy由提到了bias的概念，就是我们可以allow更多的flexibility，比如在电影打分这个context底下，可能有些user普遍对电影打分比较高，有些电影普遍比较受欢迎，我们就allow这些bias（我在这里理解的是penalize，因为Jeremy后来又提到，一个人如果喜欢动画片，那么他对动画片的打分可能尤其的高，那么这样一来，他的偏差是很大的，然而combine了这样一些偏差之后，我们其实可以得到一个unbiased用户对于电影的喜爱程度regardless of popularity or user habbits or somethine）进来，也就是在两个embedding matrices上多加一个 ‘intercept’ column，然后再去optimized。

### Training Function

* `CollabDataBunch.from_df`：协同过滤databunch的function，默认来说第一列是userID，第二列是movieID，第三列是rating。
* `learn = collab_learner(***)` will allow to choose if you want to use traditional collaborative filtering or CNN.
* `learn = collab_learner(***, score = [0, 5.5])` 加入score这个会让模型最后加入一个sigmoid layer从而constrain你的value from 0 to 5。那么为什么要设置为5.5呢，因为如果你设置成5，那么你很难达到5这个值，in order to get more accuracy. 
* `learn =collab_learner(***, wd=1e-1)`：weight decay，是regularization的一个表示，可以回想一下，在之前学推荐系统的时候，alternating least square需要去minimize的那个loss function中有个penalized term，而这个term会带有一个alpha项去控制它具体penalized多少。Generally it's 0.1, 因为Jeremy说他们做了很多实验发现0.1的时候很少出现表现不好的情况。但是呢，fastai中wd的默认值为0.001， 是个比较保守的数值，因为wd过大会让你underfit不管你训练多久，但是小一点的数值会让模型拟合的比较好，但是确实是容易overfit的，所以要realize这一点然后earlystopping。
* `learn =collab_learner(***, n_factors=40)` n\_factors就是matrix factorization中latent factor的大小，Can put the model in the for loop to select the best n\_factor.
* `learn.bias` is to ask the learner to give us the bias of user or item.
* `learn.weight` is to ask the learner to give us the weight of user or item. Usually people can do PCA to reduce the number of latent factor in order to better understand its meaning. 比如你对美国的州做embedding，然后PCA到2D，可能就会看到Embedding实际上找到了一些地理位置上的关系，将地理位置近的州放在了一个cluster中。 

#### Detail of Source Code

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc2w51r1v4j312s0dywnx.jpg)

* EmbeddingDotBias is a nn.Module. And in pytorch all layers and models are nn.Module
* 他看起来像个function，但是实际上不是，因为function require \_\_call\_\_. Take a look at [here](https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call).
* 这里的forward实际上是代替了call，当你EmbeddingDotBias\(\)的时候，python就会帮你call forward这个method。

### Affine Function

和Matri mul很相似，but more general，是多种线性函数的组合。In CNN, it is the matrix multiplication with some parameters are tied so affine function is more appropriate. 或者可以想象成，multiplication then add something \(bias\). 但是Jeremy在这里又提到，多层的Affine Function是不会达到Universal Function Approximation的，因为这相当于是多个linear function的叠加，你需要在中间加上一些Non-linear function，比如说是ReLu。

### Regularization

一个拥有很多参数的模型是非常flexible的，可以通过非常多的非线性去描述各种各样生活中数据的curvy性质，但是有人会说这样不是很容易overfitting吗？我们可以通过不用那么复杂的模型去fit呀？但是实际上通过减少parameter的个数去avoid overfit这个方法是很不可取的，因为somehow你就是需要一个复杂的模型去fit一个非常curvy的wave，但是你可以通过让这个模型不过度弯曲从而减少overfitting，而其中的方法就是正则化，对系数来做penalized。

#### L2

如果只用一个单纯的L2，可能这个值会过大，就会导致underfit，也就是过度penalized，那么我们就想要控制这个penalized的速度，也就是weight decay in fastat。

### Leaner

All of the learner in fastai create a Leaner and other parameters can be checked here.

### Loss

Loss is some function that takes the independent variable x, which is the feature and some weights to compute the predicted value and calculate the 'distance' with ground truth. Oftenly, our loss function involves weight decay coefficient and penalization. And notice that the reason why we call it a weight decay is because we take the derivative of the regularization part of the loss function, we will get a `2*wd*w` and when we are doing the gradient descent, we actually update the weight involving subtracting some of itself from last step. That's weight decay.

![image-20200220140155783](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lmzgnezj31080kiwl3.jpg)

## MINST SGD

* 这个dataset里面有50000张手写数字，但是是以flatten的形式来存贮的，所以如果要show就reshape成28x28的形式。

![image-20200220112003064](https://tva1.sinaimg.cn/large/0082zybpgy1gc3gylbpqcj31r60j8409.jpg)

* 之后我们想要用pytorch中的一些function帮助我们使用MSE，计算梯度以及其他等等，所以我们要将object从数组变为Tensor的形式。

![image-20200220112105064](https://tva1.sinaimg.cn/large/0082zybpgy1gc3gzm0q1fj31si06mwfo.jpg)

* 准备工作完成了，我们现在可以使用pytorch去帮我们做我们想要的事，并且是以mini batch的形式。

![image-20200220112146577](https://tva1.sinaimg.cn/large/0082zybpgy1gc3h0bt1fij31re09aq4n.jpg)

* Recall that pytorch当中的object都是nn.Module这个class的，所以我们想要去create一个这样的class，但是同时又想让他做一些不一样的事，那应该怎么做呢，我们就在nn.Module的基础上create a subclass. `super().__init__()` is important to superize the nn.Module original init class. Check [here](https://medium.com/the-renaissance-developer/python-101-object-oriented-programming-part-2-8e0db3ddd531) for more information on sub-class.

![image-20200220112543646](https://tva1.sinaimg.cn/large/0082zybpgy1gc3h4g6t8rj31q807cq46.jpg)

* 而上面code中，`nn.linear`帮我们做的实际上就是下面这一段中，为x加上intercept column并且和a做矩阵乘法的部分。而以上这一部分就是一个简单的logistics regression model known as a neural network without any hidden layers.

![image-20200220134219905](https://tva1.sinaimg.cn/large/0082zybpgy1gc3l2m17qwj31m407qmyh.jpg)

* 创建的nn Module需要手动告诉pytorch去run在GPU上。

![image-20200220134823159](https://tva1.sinaimg.cn/large/0082zybpgy1gc3l8v8nluj31so02kmxb.jpg)

### Cross Entropy Loss

* 接着我们想要去construct我们的loss function which is **cross entropy loss**. And then you can do the training using SGD.

![image-20200220140723605](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lsn1hlpj31w40joq6f.jpg)

* What if we try building neural network, instead of logistic regression. What we need is some hidden layers and activation layers.

![image-20200220140423281](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lpisx33j31s20bojtc.jpg)

### Adam

* Now we can actually use a optimizer to replace our mannually coded gradient descent, for example, you can do SGD with learning rate 2e-2 and get a loss curve like below.

![image-20200220141102008](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lwfwg5nj316y08etb6.jpg)

![image-20200220141115097](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lwn8j9hj30ng0e4myy.jpg)

Or we can change a optimizer which is **Adam** with learning rate still 2e-2. But now notice that the loss is diverge.

![image-20200220141236846](https://tva1.sinaimg.cn/large/0082zybpgy1gc3ly2xdnvj31r609eabj.jpg)

![image-20200220141303030](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lyixg38j30j20bcgpg.jpg)

And since we change a optimizer, we should change a learning rate for this, when we change the learning rate to 1e-3. The loss will become:

![image-20200220141356050](https://tva1.sinaimg.cn/large/0082zybpgy1gc3lzgvdb2j30uo0im79e.jpg)

Compared this with SGD loss curve, we found that the it converges faster than SGD, when ephch is 200, Adam gives us the loss down to 0.2 where SGD didn't get to this point.

#### Monentum\[[1:48:40](https://youtu.be/uQtTwhpv7Ew?t=6520)\]

如果单纯用上一步的梯度来更新这一步的parameter， 会发现这样做的速度非常慢，这时候我们引入一个动量的概念。在更新的时候，我们不仅仅根据的是上一步的梯度，但是10%的梯度然后90%是上一次的更新值（也可以是梯度，因为在第二步中，上一步的更新值就是上一步的梯度，但是当你次数变多，更新值就不再是一个单纯的梯度了）。这个时候你会发现下面动图这样方式的更新。首先红线，如果你不断的考虑上一步的步长which is 90%， 那么你的步长会越来越大，然后当你go too far，这时你的导数会和你的动量方向相反，那么你就会得到一个小一点的值，也就是小一点的步长。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc3npyqkwrg30e20aatfc.gif)

总的来说，我们的current step的步长依据的是下面这个式子，值得注意的是，上一步的步长中，包括了上一步的gradient，也包括了好几个1-alpha相乘的前几步的步长，这里的意思就是，上一步的gradient会得到更多的weight。This is called an **exponentially weighted moving average**.

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc3nu3mv4gg305500i0h3.gif)

![image-20200220152246883](https://tva1.sinaimg.cn/large/0082zybpgy1gc3nz3b6zxj31eg0j6dkd.jpg)

#### RMSProp

It's a optimizer developed by Geoffrey Hinton and the first time he introduced this is on Coursera. This time we have the exponentially weighted moving average on current step gradient squared not on the gradient update. So we have the 10% of gradient squared and 90% of the previous value 也就是上一步的步长，This is actually exponentially weighted moving average of the gradient squared.

如果我们的gradient非常小的话，那么它的平方也是一个很小的值，反之。并且在这次的update的过程中我们所做的是将原本只是 -lr\*g，增加为再除以一个sqrt of 上一步的步长 without learning rate。

所以也就是说，如果我们上一步的步长一直都很小，eg. gradient is consistently small and not volatile. 我们就需要步长变大一下去跳出这里。在这个excel的例子里面我们的intercept变化的特别慢，我们就需要加大一些步长去更新。

#### Adam

It's simple just momentum + RMSProp. 也就是把经典的SGD中的更新值中的gradient替换成exponentially weighted moving on the last updated value and devided by the sqrt of last step size without learning rate.

### Optimizer vs. Learning rate

首先先明确这两个不是一样的东西，并且作用也是不同的，Optimizer是让到达最小值的过程converge的更加快，也就是控制了我不谈及learning rate下的步长，而learning rate则是在这个基础上进一步控制了步长，从而使模型更好的停留在最优点的附近。在Jeremy的例子中，leanring rate是1的时候，尽管Adam已经让我们在10个epoch以内找到了最优的位置，但是因为learning rate过大使得我们的输出一直在附近波动，当改小learning rate的时候，结果就趋近平稳了。所以这也就显示出为什么learning rate annealing和Optimizer是共同重要的。

### Fit-One-Cycle

It's by Leslie Smith.

虽然Jeremy在前面夸了好半天的Adam和各种Optimizer，但是呢somehow，他并不tend to使用optimizer去解决问题，像这样：

![image-20200221151352996](https://tva1.sinaimg.cn/large/0082zybpgy1gc4tc5a1u7j31qq0bgjt5.jpg)

在Fastai中，我们使用的一直是based on FitOneCycle的Learner, 像这样：

![image-20200221151450666](https://tva1.sinaimg.cn/large/0082zybpgy1gc4td4soe6j31em03i0t4.jpg)

Learner呢，它take你的databutch，一个nn.Module的object，告诉它你的Loss function以及你想要print out出来的东西。我们再通过去找最好的学习率和通过freeze，unfreeze来调整模型。在这个例子里面，FitOneCycle的效果要比Adam的效果要好：

![Adam](https://tva1.sinaimg.cn/large/0082zybpgy1gc4ti66bg0j30po0e20uj.jpg)

![FitOneCycle](https://tva1.sinaimg.cn/large/0082zybpgy1gc4tio5g2dj31pk080753.jpg)

那么FitOneCycle到底做了什么呢？

![image-20200221152110881](https://tva1.sinaimg.cn/large/0082zybpgy1gc4tjqrb9gj31sg0h8q67.jpg)

来看看这两个图，左边的图我们看过是learning rate的图，从小到大，也就是从一个可能毫不make sense的starting point先走到一个make sense的parameter space，一旦进入了这个空间，你就想要学习的更加快，然后慢慢接近一个最优结果之后，你就annealing你的学习率，使得精准的达到目标。

而右边的图是动量图，我们可以看到动量和学习率的联系是，每当你学习率小的时候，你的动量就大，反之。不过其实我没有听懂Jeremy的这一段解释.... 所以我先空着。

并且Loss的画图在两者之间也是不同的，比如Adam的loss的噪音是非常大的，而fastai的loss是非常干净的一条线，due to exponentially weighted moving average of loss that fast.ai used.

## Tabular Model

在分类问题当中呢，我们想要一个loss function，能做到给出正确且high prob/confidence的预测时，损失值是很低的，相反，当你给出错误的预测但是你的confidence real high，这个情况下你的损失值应该是高的。

### nn.CrossEntropyLoss

看看一下两个图，都是算Entropy的方法，其实也就是在做matrix multiplication的， it is the same as an index lookup \(as we now know from our from our embedding discussion\). So to do cross entropy, you can also just look up the log of the activation for the correct answer.

![img](https://tva1.sinaimg.cn/large/0082zybpgy1gc4u28849cj30ex062tai.jpg)

![image-20200221153924621](https://tva1.sinaimg.cn/large/0082zybpgy1gc4u2qixgkj31e20dun4g.jpg)

不过这个情况只能work under the situation that these rows add up to one. 所以你就要去确保你的激活函数是正确的。在这里，正确的激活函数是softmax。

#### Softmax

Work well with multiple labels. This thing `softmax` is equal to [![e](https://tva1.sinaimg.cn/large/0082zybpgy1gc4u82xy5bg3008008036.gif)](https://camo.githubusercontent.com/bfe511d6658c93dcc5bca7133dedfe59b13f1455/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f65) to the activation divided by the sum of [![e](https://tva1.sinaimg.cn/large/0082zybpgy1gc4u82xy5bg3008008036.gif)](https://camo.githubusercontent.com/bfe511d6658c93dcc5bca7133dedfe59b13f1455/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f65) to the activations. That's called softmax.

![Image result for softmax](https://tva1.sinaimg.cn/large/0082zybpgy1gc4u8qtmuzj30yh0i6wl3.jpg)

![image-20200221154904930](https://tva1.sinaimg.cn/large/0082zybpgy1gc4ucrxsicj30p00gu75v.jpg)

* all of the activations add up to 1
* all of the activations are greater than 0
* all of the activations are less than 1

### Summary

所以总的来说就是，如果你有多label的数据，你想要CrossEntropy当做你的Loss，并且为了让这个work，你就用softmax作为你的activation function。对于pytorch，当你直接去调用了nn.CrossEntropyLoss时，pytorch会自动帮你在最后加上一层softmax激活层，但是有时候可能你是自定义的loss而只在背后调用了CrossEntropyLoss，所以这个时候fastai或者是pytorch不能很好地detect到你的loss，所以这个时候你的输出值可能会很奇怪，但是你要意识到可能是因为没有softmax这一层激活层的原因。

