---
description: >-
  Original Detailed Notes:
  https://github.com/hiromis/notes/blob/master/Lesson6.md
---

# Lesson 6: Regularization; Convolutions; Data ethics

## Rossmann

这个dataset值得注意的地方：

1. Training set的时间比较久远，Test set时间比较近
2. Evaluation是percentage RMSE
3. 可以使用别的dataset combine到一起来做。
4. Time Series data

   * Use RNN?
   * Actullay for statistical time series techinique, what they ususally do is only focused on the data that has the time sequence but in real life, it's impossible. 实际上会有别的有用的信息可以用。但是这些信息通常会被转换为一些别的关于date的信息，这样就可以把一些简单的时间序列问题转换为了表格问题，但是对于一些负责的时间序列data比如股票交易就不太能行。

   ```python
   add_datepart(train, "Date", drop=False)
   add_datepart(test, "Date", drop=False)
   ```

![image-20200222183120285](https://tva1.sinaimg.cn/large/0082zybpgy1gc64o0lsrrj31050u079a.jpg)

### Categorify

对于classes的preprocessing, 也就是encoding。注意这里对test set的预处理要标注是test，因为可能test中有train中没见过的label，对于这些label我们仍旧要当自己没见过。

```python
categorify = Categorify(small_cat_vars, small_cont_vars)
categorify(small_train_df)
categorify(small_test_df, test=True)
```

这样处理的之后，表格的showing方式还是一样的，但是内在里面pandas会存为数字的方式。如下图，通过`cat.categories`来查看出现的unique的label，`cat.codes`来查看number的存储方式。-1表示NaN，As you know, these are going to end up in an embedding matrix, and we can't look up item -1 in an embedding matrix. So internally in fast.ai, we add one to all of these.

![image-20200222190042671](https://tva1.sinaimg.cn/large/0082zybpgy1gc65ii63fwj31em0kwwgk.jpg)

### FillMissing

![image-20200222190623695](https://tva1.sinaimg.cn/large/0082zybpgy1gc65ohb9zej31f20jaadd.jpg)

新创建的Column，会是个boolean indicate这个对应的原数据是不是missing。Imputation的方式是用median。接下来prediction的时候就可以结合这两个column一起。

### TabularList.from

当你call任意一种TabularList的时候，你可以直接parse in一个list of preprocess that you wanna do.

![image-20200222191032414](https://tva1.sinaimg.cn/large/0082zybpgy1gc65sq1vo2j31fq0aomyx.jpg)

Notice that day, month, year should be treated as categorical variable. But sometimes and for some variables, it depends. If for this variable, the cardinality which is the number of unique label in it is not too high, it usually should be treated as categorical variable. 当然你可以都试试，看转换成哪个效果更好。

### FloatList

因为这个数据集中，y variable是整数类型，所以fastai会assume我们要做分类，所以你要告诉fastai这个是浮点集，也就是让他做回归。

### log=TRUE

it's going to take the logarithm of my dependent variable. Why am I doing that? So this is the thing that's actually going to automatically take the log of my y. The reason I'm doing that is because as I mentioned before, the evaluation metric is root mean squared percentage error.

![](https://camo.githubusercontent.com/41ae80bacb4723b8211558536d71d4d4f6bf53be/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5c74657874726d7b524d5350457d2673706163653b3d2673706163653b5c737172747b5c667261637b317d7b6e7d2673706163653b5c73756d5f7b693d317d5e7b6e7d2673706163653b5c6c656674285c667261637b795f692673706163653b2d2673706163653b5c6861747b797d5f697d7b795f697d5c7269676874295e327d)

Neither fast.ai nor PyTorch has a root mean squared percentage error loss function built-in. I don't even know if such a loss function would work super well. But if you want to spend the time thinking about it, you'll notice that this ratio if you first take the log of y and y\_hat. then becomes a difference rather than the ratio. In other words, if you take the log of y then RMSPE becomes root mean squared error. 所以之后就用default的RMSE就好了。或者可以这么理解，我们的y其实是long tail，in the most of the time, 我们想要让让他变成正态一些的分布，又或者说我们想要在意的是比值差异，而不是绝对值差异等等，所以either使用log with RMSE或者用直接用RMSOE。

### y\_range

在协同过滤中，我们有提到最好是通过sigmoid使得最后的输出范围在一个我们想要的范围内，这里是类似的，但是值得注意的是，我们的range也要取对数。并且和之前一样，我们的range要大一点这样才能让我们也有可能达到这个真实最大值。

![image-20200222194100832](https://tva1.sinaimg.cn/large/0082zybpgy1gc66og1e8lj31rs06y3zv.jpg)

### Tabular\_learner

![image-20200222194447645](https://tva1.sinaimg.cn/large/0082zybpgy1gc66sb94gjj31po03e74t.jpg)

对于表格来说，目前来说其实最好的模型就是fully connected layers with combination of matrix multiplication and activations。 这其中，我们可以看到layer的部分，有1000个activation input和50个activation output，可以计算出这个模型有1000_50这么多的参数，对于我们的数据来说其实是太大了，所以会overfitting。所以要避免就是Regularization， \*not to reduce the number of parameters_。fastai默认操作是wd，而我们在这个情况会需要更多。

### Dropout

So we're going to pass in something called `ps`. This is going to provide dropout. And also this one here `embb_drop` - this is going to provide embedding dropout. See the [paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) for more details. 来看看下面这个图，This first picture is a picture of a standard fully connected network，而每一条箭头都是activation value和参数相乘，然后汇聚在一起的意思就是做相加，结果就是上一层的activation和参数相乘再求和。而dropout就是我们随机丢掉**激活值**！对于每个mini-batch我们丢掉的激活值是不一样的，那么丢掉多少呢？每一个丢掉的概率一般来说是0.5，并且呢我们一般不丢掉input data，也就是丢掉hidden layer中的一些中间计算值，而且这样丢掉就意味着前面计算的值也会被丢掉，这样somehow就抹去了神经网络一部分的记忆，去防止他们死记硬背从而过拟合。我觉得Hinton太牛逼了。当然过度dropout会让你underfit，所以你要调节不同的activation layer with different dropout probability.

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc675e22loj313m0tykd6.jpg)

所以fastai基本上每一个leaner都包含了这个ps的参数，也就是每一层的丢弃比例。So parsing a list will apply the prob to the layers or parsing a int that will apply to each layer. Sometimes it's a little different. For CNN, for example, if you pass in an int, it will use that for the last layer, and half that value for the earlier layers. We basically try to do things represent best practice.

另一个值得注意的地方就是，我们只想在training的时候做dropout，因为dropout是想泛化，我们在做inference的时候不想要泛化，我们想as accurate as possible，所以在test time的时候我们不做任何的dropout，那么会出现这样一种情况，假如你的p是0.5，你有一半的element会被dropout when training, but when you are doing the inference, the number of params are actually twice of your training. And maybe you want to do the weight \* p when inferencing. Anyway, pytorch doesn't perform dropout when inferencing.

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc67slq3kej30x40bqac0.jpg)

### Dropout on Embedding Layers

For continuous variables, that continuous variable is just in one column. You wouldn't want to do dropout on that because you're literally deleting the existence of that whole input which is almost certainly not what you want. But for an embedding, and embedding is just effectively a matrix multiplied by a one hot encoded matrix, so it's just another layer. So it makes perfect sense to have dropout on the output of the embedding, because you're putting dropout on those activations of that layer. So you're basically saying let's delete at random some of the results of that embedding \(i.e. some of those activations\).

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7e2bqnumj31500u0jz2.jpg)

### BatchNormalization

Works for continuous variable, below the architecture shows that 16 for its BN layers since we have 16 continuous variables. 下图关于BN的论文中也做了实验，发现用了BN之后，converge得非常快。那到底什么是BN呢？Jeremy说在早期一点关于BN的paper会说BN is accelerating training by reducing internal covariate shift. 但是后来发现不是通过这一原因去加快converge的，但是呢根据第二章图可以看到加了BN之后的模型的损失会变得更加稳定一下，不会那么bumpy，那么在这个基础上我们就可以稍微增大学习率去学习，因为在模型稳定的基础上我们可以不用过于担心增大学习率会把我们带出optimal space。那再来看看BN是怎么做的，下图的右边，从上一层拿到activation values归一化只有，通过乘上一个bias，加上一个bias来做BN，这两个参数也是可以调节的，也是在GD中更新的。但是具体是怎么做到优化的呢？Assume我们现在的深度学习网络是这样的

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7epa9ebxg305h00i0lr.gif)

我们也许想要我们的ouput在1-5之间，但是somehow最后激活层给我们的值会让结果有一些off，可能是在-1到1之间，这时候我们想要去调节这个结果，可以通过改变scale和mean，如何做呢？直接在网络内部调节其实是有一定困难的，因为里面的参数或者说是调节的手段非常的多并且都是相互影响的，所以一个比较直接的方式就是把网络变成这个样子：

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7esqph7zg307800i0qq.gif)

We added 2 more parameter vectors. Now it's really easy. In order to increase the scale, that number g has a direct gradient to increase the scale. To change the mean, that number b has a direct gradient to change the mean. There's no interactions or complexities, it's just straight up and down, straight in and out.

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7e3y685mj315c0lsk7i.jpg)

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7egjkrsej31260nwtjk.jpg)

在代码的实现当中，我们为每一个continuous variables都建立了一个BN层，去和下面模型结构中的16对应，而且我们也能发现在BN层我们会有一个动量的参数，这个动量的参数实际上是我们在求mean和sd时做的exponentially weighted moving average，原因是如果每个Batch之间的mean和sd相差过大也会导致训练不好。如果动量比较小，那么意味着批量之间的差异会比较小，那么正则化效果也会比较小，反之。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7e3lg8blj30re08u44o.jpg)

### Model Architechture

1116是对应的Store这个categorical variable的个数，后面是你embedding的column数，default tends to work well but you can still define your own.

![image-20200222194438692](https://tva1.sinaimg.cn/large/0082zybpgy1gc66s6ngz4j30yk0u0gtg.jpg)

### Data Augmentation

也是正则化的一个方式。 在增强的时候，要做的两件事，是确保你做数据增强之后你的照片依旧清晰，可以铜鼓打印出照片来查看，其次是增强之后，处理过的照片也能作为一类独特的数据来训练模型。Warping是个很棒的增强手段，让你的图片像有从各个角度拍摄出来的一样，fastai在这个方面是pioneer，但是一致的注意的是，在查看的时候，可以把border的填充关闭，从而看到更直接的效果`padding_mode=padding_mode,`设为0，但是训练的时候，填充方式默认reflection比较好，更和现实世界的图片相似。

## Pet Revisit

我们现在想要手动生成一个热力图，从而显示出CNN focus在哪一个部分。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7fzkrjuoj3075070abz.jpg)

### Convolution

之前看过的这幅图，Jeremy也提到过，是全连接层的一个非常好的体现。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7g0mie1oj310l0foti5.jpg)

可以来看看这个[网站](http://setosa.io/ev/image-kernels/%20)，上面这个3x3的matrix是convolutional kernel，然后对于移动到的像素点值，做element wise的乘积和，得到一个值，这个过程就是卷积的过程。这个kernel无法计算最外边框，那么对应过去就填成了0，也就是黑色。

![](https://github.com/hiromis/notes/raw/master/lesson6/31.png)

也可以另外一种理解，首先可以把PQRS from output matrix看做是带有一个normal bias b的来自原matrix的线性组合。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7gbowkt2j30a2095t8w.jpg)

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7gbwnudgj306003eaa0.jpg)

而上面这个情况可以写成下面这个神经网络的形式，灰色的线表示的是0.

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7gd401hdj309u08775i.jpg)

而上述的过程，又可以写成以下这个样子，只不过卷积核的部分很多都变为0

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc7gf9rzsgj30jw05jwf3.jpg)

Here is our list of pixels in our 3x3 image flattened out into a vector, and here is a matrix vector multiplication plus bias. Then, a whole bunch of them, we're just going to set to zero. So you can see here we've got:

[![img](https://tva1.sinaimg.cn/large/0082zybpgy1gc7ggefrvkj305j00mt8h.jpg)](https://github.com/hiromis/notes/blob/master/lesson6/37.png)

which corresponds to

[![img](https://tva1.sinaimg.cn/large/0082zybpgy1gc7ggf76nej301v01s3ya.jpg)](https://github.com/hiromis/notes/blob/master/lesson6/38.png)

In other words, a **convolution** is just a matrix multiplication where two things happen:

* some of the entries are set to zero all the time
* all of the ones are the same color, always have the same weight

So when you've got multiple things with the same weight, that's called **weight tying**.

当然matrix multiplication（也就是以上的方法）来实现卷积是可以的，但是很慢，

#### Padding

![image-20200224154122635](https://tva1.sinaimg.cn/large/0082zybpgy1gc8azq59amj30h607mt93.jpg)

3x3的kernel在3x3的matrix上只能对应出一个值，所以为了算出更多的值，就在最外层填充一圈数值，（pytorch一般是填充0，但是fastai如果情况允许，一般使用的就是reflection padding，但是在simple CNN，fastai还是用的0填充，而且对于大图来说不matter so much）这个过程就是Padding。

回到卷积上来，如果我们输入是一个彩色图片，那我们就应该有rank 3 tensor，正常来说，我不想用同一个kernel去对三层做卷积，因为假如说，你想要做一个青蛙的检测器，那你一定想要emphasize more on the green color which means that more activations on the green thean we would on the blue. 所以我们要一个3x3x3的kernel，将27个值相乘相加得到一个值。为了得到原大小size\(5x5\)的matrix，我们加一层padding就能得到一个5x5的output matrix，但是这里只有一层呀，我们对一层matrix结果可以做的非常有限，所以为了得到更多层的output，**我们建立多个kernel，将output stack到之前的matrix上**。

Generally speaking, our input is HxWx3, and we ususally define 16 kernels and have 16 dimensions ouput. 那么得到的这16个channel的ouput会包含比如 how much left edge was on this pixel, how much top edge was in this pixel, how much blue to red gradient was on this set of 9 pixels each with RGB的这些简单的几何信息。之后你可以做同样的事情，但是值得注意的是我们想要模型越来越深的时候能识别的object更加具体，那么就有必要增加channel的个数，所以一般来说，后面layers的kernel数量会比前面的要大。

![](https://github.com/hiromis/notes/raw/master/lesson6/whiteboard.gif)

#### Stride

卷积之后，matrix大小算法：`input size - kernel size + stride = output size`

要注意的是Conv2d和Conv1d，这里的数字和stride没有关系，而是是否对高做卷积，也就是Conv1是对一个序列, 而Conv2d是对一个图片, 不考虑channel个数, 所以实际上来说是一个3d的. 下面这个link

[https://zhuanlan.zhihu.com/p/95058866](https://zhuanlan.zhihu.com/p/95058866)

很多时候我们为了避免过大的内存占用 since channel number is increasing。我们通常跳过两个像素去做kernel的multiplication，比如从\(2,2\)-&gt;\(2,4\)-&gt;\(2,6\)...

这个过程是stride为2的卷积。过程还是一样的，只不过要跳过两个pixel。所以output是H/2xW/2x32。

回到pet的notebook上，我们图片的大小是352x352, 从architecture上来看，第一层的stride为2的卷积，output是一半大，并且有64层activation通道，而且后面把像素降到了88， 44，并且越来越小。

![image-20200224162533579](https://tva1.sinaimg.cn/large/0082zybpgy1gc8c9n632ej30kp0f4tad.jpg)

而channel的个数越来越多。End up with \[512, 11, 11\].

![image-20200224164610741](https://tva1.sinaimg.cn/large/0082zybpgy1gc8cv3qo7uj30mv0kwacb.jpg)

从下图也可以看到，一上来就直接用了7x7的kernel大小，part2会解释为什么在浅层的时候会需要一个大一点的kernel\_size

![image-20200224162739780](https://tva1.sinaimg.cn/large/0082zybpgy1gc8cbu5ofij30ss0evdjh.jpg)

### Mannully Create Convolution

我们现在想要对这只狗建立一个kernel，以下所建立的kernel我们expect它能识别右下角的一些object，之后为了match 3 channel我们相当于复制了三遍这个kernel变成了一个3x3x3的kernel。意味着我们想对每个颜色做的事情是一样的。怎么去interpret这个\[1, 3, 3, 3\]的shape呢，最后两个3是这个kernel是3x3的，我们现在有3个，也就是第二个3，第一个1是表示这三个3x3的kernel整合为一个tensor。

![image-20200224190031508](https://tva1.sinaimg.cn/large/0082zybpgy1gc8gr16o58j30u00wydn4.jpg)

而对于这个图片的维度处理呢，remember在pytorch当中，所有的图片是以批量出现的，就算只有1张图片，我们也要增加一维向量去indicate这个图片batch有多少张。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gc8gy1mppsj30u00vfgvg.jpg)

### Heatmap

#### Pooling

想象一下，我们最终想要一个37长度的probability vector，假如我们的上一步output是一个11x11x512的matrices，那我们要做到是对11x11x1取一个均值，重复512次，得到一个长度为512的均值vector，然后让这个vector和一个512x37的matrix相乘得到37的output。取均值的过程就是average pooling。而对应到model里面的就是AdaptiveConcatPool \(More detail in Part2\)。output size也就是一个面一个均值。

![image-20200224191756745](https://tva1.sinaimg.cn/large/0082zybpgy1gc8h90v5naj31cm0d877b.jpg)

在plot heatmap之前，我们要搞清楚，我们这个heatmap的目的是为了知道CNN focus在什么地方才让它非常confidence的认为了这个是个缅因猫，我们可以这样理解，confidence的认为是缅因猫也就意味着我们的probability对应缅因猫这个label非常大，在上一层也就要对应着相对更高的值，这个值也就是从之前的weighted sum through matrix multiplication得到的，之前我们是对512个11x11取了均值，把这512个phase理解为对于这个图片的特征，比如耳朵有多尖，尾巴有多长，猫有多长等等，所以make sense我们要对512个phase取均值来看每个特征的比重，但是呢在heatmap里，我们实际上是想看到这些area有多activated，所以我们就take average across the 512 to get a 11x11 matrix。`avg_acts`就是这个11x11的average matrix across 0 dimension which is th channel dimension.

#### hook

那上面这个avg怎么来的呢，用hook，去链接到pytorch内部的结构。因为在做forward的过程中，我们也会得到512x11x11这个matrix的计算，而不是仅仅直接得到了最后的activation的output，而我们想hook into pytorch forward part，然后告诉pytorch把我要的那个512x11x11存下来给我。那这个matrix被称作是part of convolutional part of model。而在平均汇聚之前的所有层，不包括输入层，都称作为convolutional part。回想一下transfer learning，我们是替换掉了卷积部分的最后几个，对于fast.ai的存储来说，fast.ai的首位一定是模型的卷积部分，也就是m\[0\]。所以我们想要`hook_output(m[0])`, 但是之后要把hook给去掉，因为这会占据很多内存。

![image-20200226095934986](https://tva1.sinaimg.cn/large/0082zybpgy1gcaccq7u61j31tk0pogq9.jpg)

## Ethics

