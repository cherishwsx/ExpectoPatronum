---
description: >-
  Original Detailed Notes:
  https://github.com/hiromis/notes/blob/master/Lesson7.md
---

# Lesson 7: Resnets from scratch; U-net; Generative \(adversarial\) networks

## Jupyter Notebook Tips

如果你想释放你GPU的内存，instead of restarting the kernel, you can first set the object that you think takes up the memory space and then do garbage collection in python. 但是如果你用Nvidia SMI去check的话，可能还是会看到内存被占用，因为PyTorch会存一下cache之类的东西，but it can be used anyway。

```python
learn_gen=None
gc.collect()
```

## Fast.ai in App

Food Classifier: [https://reshamas.github.io/deploying-deep-learning-models-on-web-and-mobile/](https://reshamas.github.io/deploying-deep-learning-models-on-web-and-mobile/)

## MNIST CNN

### Read Data

因为MNIST是黑白数字，所以convert\_mode设置成L。读成一个1x28x28的image。再强调一下为什么是3D tensor，因为conv2D works with 3D tensor。

```python
il = ImageList.from_folder(path, convert_mode='L')
```

### Split Data

也可以不split，但还是要tell fastai to do no split。并且train和valid是有label的，test是没有label的。

![image-20200226133224467](https://tva1.sinaimg.cn/large/0082zybpgy1gcaii3ugqfj31t00ign03.jpg)

### Transformation

因为数字本身不太能旋转和反转，而且本身pixel就比较小，如果zoom会更加不清楚，所以就是用padding和cropping做transformation。用\*去表示把这两个transformation放在一起，后面的list是空意味着我不要对validation做transformation。

![image-20200226133454393](https://tva1.sinaimg.cn/large/0082zybpgy1gcaikpd47qj31r60l2dhq.jpg)

### Create DataBunch

不使用transfer learning的时候就不需要用ImageNet的mean和variance。

![image-20200226133829633](https://tva1.sinaimg.cn/large/0082zybpgy1gcaiofhvpuj31qe0a0myg.jpg)

### Check Data

我们得到了一个data bunch，在data bunch里有一个我们已经看过的data set。值得注意的是，现在这个训练集有了数据增强，因为我们做了变形。`plot_multi`是一个fastai函数。它会画出为这个行和列里组成的网格里每一个元素调用这个函数的结果。这里，我的函数只是取出训练集的第一个图片，因为每次你从训练集里取出一些东西，它会从硬盘里加载它，实时对它做变形。人们有时问创建了多少图片变形的版本，答案是无限个。每次我们从数据集里取出一个东西，我们实时做随机变形，所以基本上每一个都是有点不同的。所以你可以看到，如果我们画很多次，8都在有点不同的位置，因为我们做了随机边框（random padding）。也可以是使用show\_batch的方法。

![image-20200226134804853](https://tva1.sinaimg.cn/large/0082zybpgy1gcaiyesfckj30l30s1gn0.jpg)

### Basic CNN with BatchNorm

手动创建CNN！运用pytorch，可以自己创建自己的神经网络层。这里，所有的卷积的核的大小（kernel size）是3，步长（stride）是2，边框（padding）是1。

Recall that 每次你做卷积时，它先跳过一个像素，每次跳两步。这意味着，每次完成一个卷积，它会把网格的尺寸减半。而从7到4是因为python做除法的性质。

ni是你的输入通道的size，nf是你的输出size，而上一层的nf和下一层的ni得保证一致。最后flatten at the end是为了把输出的10x1x1的tensor变为一个长度为10的向量。

![image-20200226135256198](https://tva1.sinaimg.cn/large/0082zybpgy1gcaj3gpo2qj30m10ast9w.jpg)

### Learner

然后就是general stuff when training the model, can test one batch only then load your one batch into the model and lr\_find。

![image-20200226135844363](https://tva1.sinaimg.cn/large/0082zybpgy1gcaj9ic5a2j30jp0k7400.jpg)

### Refactor

想上面那样卷积层，BN层和ReLu的一个这样的组合，fastai已经帮你打包好了。叫做conv\_layer。x下面这个神经网络是和我们之前的表示是一样的。

![image-20200226140510088](https://tva1.sinaimg.cn/large/0082zybpgy1gcajg6u7boj30lp06ywez.jpg)

那应该怎么去improve呢，可以尝试将神经网络构建的更深，比如在每一个Conv2d后面加上一个步长为1的卷积，这样是不会改变的你output channel size的。

#### ResNet

或者我们来考虑一下[ResNet](https://arxiv.org/abs/1512.03385)结构。ResNet是微软研究院的Kaiming He和他的团队研发出来的，当他们用56层的卷积层，也就是我们刚刚在讨论的一模一样的层只不过增加了一些Conv1d，在Cifar10上做实验的时候，发现56层的卷积层比20层的卷积层效果更差，很神奇！

![image-20200226142059523](https://tva1.sinaimg.cn/large/0082zybpgy1gcajwnir51j30h108tgmv.jpg)

所以这个同学他就觉得好奇怪啊为什么呢，我不服！我要用一样和56层一样的网络结构，做出更好的效果！于是呢，他就改成了这样，每次卷积。他把这两个卷积的输入和这两个卷积的结果加到一起。

![image-20200226142112762](https://tva1.sinaimg.cn/large/0082zybpgy1gcajwuxrohj30e107874q.jpg)

上图是一个ResBlocks

换句话说，不再是用：

![img](https://tva1.sinaimg.cn/large/0082zybpgy1gcajvby693j306500i742.jpg)

而是改成用：

![img](https://tva1.sinaimg.cn/large/0082zybpgy1gcajvd420sj307100igle.jpg)

他的观点是56层的卷积网络至少应该比20层的好，因为它可以把除了前20个层以外的，conv2和conv1的层的权重全部设成0，这样X（输入）可以直接穿过去。 这个叫**identity connection**，它是identity函数，没有做任何事。它也被叫做**skip connection**。 这个ResBlocks的概念很棒，因为在以前的CV研究中，都没有用到这个blocks，之后会有更多的突破。

但是是为什么ResBlocks能带来好的效果呢？和BN一样，看看这个图，是个参数空间，z方向是loss，。左边是没有用ResBlocks带来的skip connection，右边是使用了的。很明显，右边的loss更加平缓让我们有更大的空间去调大我们的学习率。Paper is [here](https://arxiv.org/abs/1712.09913)

![image-20200226142549458](https://tva1.sinaimg.cn/large/0082zybpgy1gcak1opk5cj30gg0cs792.jpg)

#### ResBlocks

在代码里，按照刚刚讲过的，我们创建了一个ResBlock。我们创建了一个`nn.Module`，`conv_layer`（记住，一个`conv_layer`是Conv2d、ReLU、batch norm的组合），我们创建了两个这样的东西，然后在forward里，我们运行`conv1(x)`，然后对它运行`conv2`，然后添加`x`。

![image-20200226142905802](https://tva1.sinaimg.cn/large/0082zybpgy1gcak52ygemj30jq05kq3b.jpg)

所以我们可以将之前全是conv\_layer的神经网络改写成

![image-20200226144434812](https://tva1.sinaimg.cn/large/0082zybpgy1gcakl74wqpj30k5089dgc.jpg)

而且我们甚至可以进一步refactor：

![image-20200226144551969](https://tva1.sinaimg.cn/large/0082zybpgy1gcakmixfjrj30kn06gt97.jpg)

如果你在尝试新的架构，持续重构可以让你更少出错。很少人这样做。你看到的大多数研究代码都很笨重，这样经常会出错，不要这样做。你们都是coder，用你们的编程技能让事情保持简单。

#### Learner

然后你再接着learner，和之前的步骤一样。

![image-20200226144649577](https://tva1.sinaimg.cn/large/0082zybpgy1gcaknj44ipj30l20domym.jpg)

这个model是我们literally built from scratch，依旧拥有很好的准确率。和前几年前的SOTA比较起来。

![image-20200226144802523](https://tva1.sinaimg.cn/large/0082zybpgy1gcakos3rz0j30ei043t91.jpg)

![image-20200226144823529](https://tva1.sinaimg.cn/large/0082zybpgy1gcakp5i6boj30q60fm7av.jpg)

#### fastai ResBlocks

我们现在知道了ResBlocks是个非常有用的东西，并且有很多Library都在尝试着对他进行更多优化，从而运行的更快，同样的，fastai也帮你做了一个这样的implement。fastai中的ResBlocks使用MergeLayer和SequentialEx实现的。现在可以这么解释，可以看到MergeLayer里面的forward中有`x+x.orig`这个`x.orig`就是SquentialEx中来的。它和fastai里的sequential拓展类似。它就像一个普通的sequential模型，但我们把输入保存在`x.orig`。所以这个`SequentialEx`, `conv_layer`, `conv_layer`, `MergeLayer`做的事情和`ResBlock`一样。你可以用`SequentialEx`和`MergeLayer`很简单的创建你自己的ResNet block变体。

![image-20200226144349400](https://tva1.sinaimg.cn/large/0082zybpgy1gcakqqa31wj30lj07z400.jpg)

#### DenseNet and DenseBlocks

当你创建MergeLayer时，你可以选择设置`dense=True`，这会发生什么？如果你这样做，它不会运行`x+x.orig`，它会运行`cat([x,x.orig])`。换句话说，不再用加法，它做了一个concatenate（连接）。你的输入进入了Res block，当你用concatenate替代相加时，它不再调用Res block，它调用的是一个dense block。它不再叫ResNet，它叫DenseNet。如下图：

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcaky0r9gvj307e07ygmk.jpg)

可以看到，使用dense block，它变得越来越大，有意思的是，原来的输入还在这里。实际上，不管走了多深，原始的输入像素还是保存在这，原始的第一层特征还在，原始的第二层特征还在。可以想象，DenseNet很耗内存。有方法可以处理这个。时不时的，你可以使用一个普通的卷积，它会把你的通道降下来，但它们会耗费内存。但是，它们有很少的参数。所以，处理小数据集时，你应该试下dense block和DenseNet。它们在小数据集上的效果会很好。

因为它能一直保存这些原始的输入像素，它在分割问题（segmentation）上效果很好。因为对分割来说，你需要能重新组织图片的原始图像，所以保存所有这些原始像素非常有帮助。

## U-net

现在来谈谈Image Segmentation，ResNet中提到的Skip Connection对于图像分割有很大的作用。我们现在尝试用ResNet去尝试一样更新U-net结构。我们之前在Camvid上其实也用到过Unet，并且这个Paper上的SOTA成绩是91.5%。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcamkpkc7lj318m0otkdz.jpg)

而Jeremy用ResNet去更新的U-net达到了94.1%。

Recall一下之前的Notebook，为了用颜色标记出这是一个行人，这是一个骑自行车的人，它需要知道这是什么。它需要真正知道行人是什么样的，它需要准确地知道行人在哪，知道这是行人的胳膊，不是他们的购物篮的一部分。要完成这个任务，它需要真正的理解图片的很多内容。对于我们最好的performance的model时候，已经非常精确了，人眼已经区分不出错误的分割像素了。（Segmentation is a more pixel level task）。

![image-20200226155524256](https://tva1.sinaimg.cn/large/0082zybpgy1gcammwl0a5j30pv0npgy2.jpg)

我们得到这个结果的原因是因为，我们用了迁移学习！这个Unet learner使用了resnet34为basic的迁移学习。

### Downsampling \(Encoder\)

看看下面这个Unet的结构，我们使用了ResNet34，并且从一个572x572的一个超大图片开始，在经过一系列的conv2d之后，max pool成了一个128channel的，size down成一半的284的layer。然后就是一系列的重复最终成为了直到降到28x28，有1024个通道。Notice that 在这个原始的论文里，它们没有添加padding。所以它们每次做卷积时，会在每个边丢失一个像素，解释了572到570到568这里的变化。而这一系列，也就是左半边被称为是downsampling，在下面这个learner里是用ResNet34实现的。So you can see that the size keeps halving, channels keep going up and so forth.

![image-20200226155849830](https://tva1.sinaimg.cn/large/0082zybpgy1gcamqgczgoj30pg01wq2z.jpg)

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcamrg4zecj318e0mvqep.jpg)

### Upsampling/Deconvolution \(Decoder\)

现在正常size的图片已经被缩小了，在我们这个notebook的数据集里面呢，我们是从224x224变到了7x7，那我们又怎么从这个7x7变回到224呢？这个过程是通过stride half conv，被称为deconvolution，或者是transpose convolution实现的。

There is a fantastic paper called [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285.pdf) that shows a great picture of exactly what does a 3x3 kernel stride half conv look like. If you have a 2x2 input, so the blue squares are the 2x2 input, you add not only 2 pixels of padding all around the outside, but you also add a pixel of padding between every pixel \(dilated convolution?\). 但是呢单纯的增加这些白色的pixel做了很多无用功，比如你在角落做卷积的时候，你的像素基本上都是白色，那你 就是白做功，又比如你在移动的过程中，有时候take了一个有效像素点，有时候用了两个，这样相当于是你将不同的信息放入了卷积中，但是does not make sense to throw away information。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcan7sailvj314e0kh152.jpg)

* Nearest Neighbor Interpolation

那假如我们这么做，on top of this, I could just do a stride 1 convolution, and now I have done some computation and it gets a mixture of A's and B's which is kind of what you would want and so forth.

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcanl9cp6uj30dd04tglf.jpg)

* bilinear interpolation

就是不再把A复制到所有不同的格子，而是取附近格子的加权平均。对于红色格子a来说，它周围有3个A，2个C，1个D和2个B，基本上你可以用一个加权平均。双线性平均，去填充这个a值。它是一个很标准的技术。每次你在电脑屏幕上看一个图片，改变它的大小，就是在做双线性插值。所以你可以做一个双线性插值，然后做一个步长是1的卷积。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcanz6up7oj304r04st8k.jpg)

* Pixel Shuffle

fastai库实际上做的是**pixel shuffle**，也叫**sub pixel convolutions**。它不是特别复杂，但今天没时间讲它。**它们的思路都是一样的。所有这些东西都是让我们做一个卷积，来把尺寸加倍。**

但是单纯的使用upsampling，其实效果是不好的，因为你从一个28x28图想要还原到572x572本身来说就是很难的。So you tended to end up with these things that lack fine detail. 那怎么去解决这个问题呢？

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcao6lhr30j30po0h2gnv.jpg)

Olaf Ronneberger et al. 做的是添加一个skip connection，一个identity connection，但不是添加一个跳过两个卷积层的skip connection，他们添加了灰线这里的skip connection。换句话说，他们添加了从下采样过程到上采样过程的相同尺寸部分的skip connection。他们没有相加，这就是为什么你可以看到这里白色和蓝色方块挨在一起，他们没有相加，而是做了连接（concatenate）。这就像dense block，但skip connection跳过越来越多的架构，所以在这里（顶部灰箭头），你基本上把输入像素放到了最后几层里做计算。这很简单地解决了分割任务里的细节问题，因为你基本上拿到了所有细节。缺点是，我们对这些更多的信息没有做太多的computation（右上角），只有四层。你最好能在这个阶段，做了所有必要的计算来识别出这是一个骑自行车的人还是一个行人，但是你可以在上面加上一些东西来判断这个像素是他们鼻子的末端还是树的一部分。这个效果很好，这就是U-Net。

### Encoder Source Code

下图是fastai的Unet code, 我们从encoder开始，encoder也就是左边画圈圈那里，在我们这个例子里面也就是个ResNet34的网络。

所以，我们的U-Net的`layers`是一个encoder，然后batch norm，然后ReLU，然后`middle_conv`（它只是`conv_layer`，`conv_layer`）。记住，在fastai里`conv_layer`只是一个conv，ReLU，batch norm 。所以，这个middle\_conv是最底部的这两步（蓝色框框部分）。

接着，我们遍历这些索引（`sfs_idxs`）。这些索引是什么？这些是每个步长是2的卷积出现的层数，我们把它放进一个索引的数组里。然后我们可以遍历它，我们可以对每个这样的点创建一个`UnetBlock`，告诉我们有多少个上采样通道，有多少个cross connection。这些灰线叫cross connection，Jeremy这么称呼的。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcaoiv9kcpj31580mqe7l.jpg)

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcaojq86ocj30sx0h3n7a.jpg)

在fastai里面，little tricks有

* pixel shuffle
* ICNR
* 还有另外一个Jeremy上周做的方法，不仅是取这些卷积的结果，传递它，还结合了原始输入的像素甚至更加finer，做了另外一个cross connection。这就是这里的`last_cross`。你可以看到我们添加了一个原始的输入到`res_block`（你可以看我们的`MergeLayer`）。

  ![](https://tva1.sinaimg.cn/large/0082zybpgy1gcaosf522yj31580mq0yu.jpg)

  `UnetBlock`要在每个下采样点存储激活值，实现这个的方式，就是我们在上节课学的，使用hook。我们把hook放到ResNet34里，每当有一个步长是2的卷积时存储激活值，这里你可以看到，我们取这个hook（`self.hook=hook`）。我们在这个hook里取出存储的值，我们只是做`torch.cat`，所以，我们连接上采样卷积（`up_out`）和hook的结果，这是我们做过batch norm和两次卷积的结果。这里最后是个DenseBlock，可以尝试改成ResBlock。

![](https://tva1.sinaimg.cn/large/0082zybpgy1gcaoi9ya5xj314n0g84h5.jpg)

### Question

* 在DenseNet里，当图片/特征图的尺寸在每层间是不同时，怎样把每层连接起来？

  如果你有一个步长是2的卷积，你不能用DenseNet。在DenseNet里，你会做dense block，增长，dense block，增长，dense block，增长，这样你得到了越来越多的通道。然后你做一个没有dense block的步长是2的卷积。然后你再做几个dense block。

  实际上，DenseBlock不会保留你所有的信息，每到一个Stride 2的卷积的时候就会丢失一些，对于这些bottlenecking layers，会有别的方法可以处理，比如重新Reset来重新选择我们想要的channel个数，从而达到可以控制我们的memory。

* 对什么样的问题，你不使用U-Net？

  U-Net用于你的输出的大小和输入的大小接近的时候。如果这样的分辨率对输出是不必要的、没有用处的，那就不需要用cross connection。是的，各种生成模型。分割也是一种生成模型。它生成了原始图片的标记遮罩图片。所以，大概所有你希望输出的分辨率和输入的分辨率一致的情况，都要用U-Net。显然，对分类器这样的东西，没什么用。在分类器里，你只需要下采样过程，因为你最后只需要一个数字，来表示它是一只狗、还是一只猫、还是什么种类的宠物。

## Image Restoration

对于图片修复来说，我们相当于就是将图片从质量差的变为质量好的，当然会有很多升级版本可以做：

* take a low res image make it high res
* take a black-and-white image make a color
* take an image where something's being cut out of it and trying to replace the cutout thing
* take a photo and try and turn it into what looks like a line drawing
* take a photo and try and talk like it look like a Monet painting

因为我们要做图像修复，所以我们需要一个又有好的图片和坏的图片的数据集，最简单的方法，也就是这里的做法就是将好的图片变差。

### Crapiffy

在我们的例子里，变差的做法是：

* 打开图片
* 调整大小，变成小的96x96分辨率，使用双线性插值
* 然后取一个10到70随机数 （quality 10是非常差的，quality 70还可以）
* 把这个数画到一些随机的位置
* 然后用这个随机数作为JPEG画质保存图片

```python
from PIL import Image, ImageDraw, ImageFont
def crappify(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 96, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    w,h = img.size
    q = random.randint(10,70) 
    ImageDraw.Draw(img).text((random.randint(0,w//2),random.randint(0,h//2)), str(q), fill=(255,255,255))
    img.save(dest, quality=q)
```

![image-20200228103553402](https://tva1.sinaimg.cn/large/00831rSTgy1gccon4o9tlj30ky0j7gxf.jpg)

选择如何去Crappify你的图片是一个很重要的过程，如果你想为黑白图片着色，你要把它变成黑白的。如果你想补全被裁剪掉一块的图片，就添加一个大黑块。如果你想处理有折痕的旧照片的扫描件，就试着想想在图片上添加印刷墨点和折痕的办法。（你的model只能学会你crappify里面的东西）

### Parallel Crappify

将图片一张一张变差是很要takes a long time的。所以我们会利用fastai中的parallel function。只需要输入你的functioin和你想要作用在这个function上的list。

```python
il = ImageItemList.from_folder(path_hr)
parallel(crappify, il.items)
```

### Generator

我们现在想要把左边的图片输入一个模型然后输出一个右边的图片。那么U-net with ResNet34 architecture and using transfer learning is very appropriate.

* Why transfer learning? 因为如果我们想去掉这个46，你需要知道这可能是什么，你需要知道这是一个什么的图片。不然，怎样知道它应该是什么样子？所以我们要用预训练模型，它知道这些东西是什么。
* MSELossFlat\(\): MSELoss是比较原图和生成图片的像素点距离的MSE。而且它takes的是两个vectors所以我们就flatten图片像素成两个vector。
* blur, norm\_type and self\_attention is important but will be discussed in the part 2.

```python
wd = 1e-3
y_range = (-3.,3.)
loss_gen = MSELossFlat()
def create_gen_learner():
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=y_range, loss_func=loss_gen)
learn_gen = create_gen_learner()
```

之后开始训练，这个时候我们还是训练 with freezed的，freeze的部分是down sampling的部分，也就是我们ResNet34所在的位置。这个`pct_start`的解释可以看看[这里](https://forums.fast.ai/t/what-is-the-pct-start-mean/26168)。

it is the percentage of overall iterations where the LR is increasing.

So, given the default of 0.3, it means that your LR is going up for 30% of your iterations and then decreasing over the last 70%.

```python
learn_gen.fit_one_cycle(2, pct_start=0.8)
```

之后再train with unfreeze

```python
learn_gen.unfreeze()
learn_gen.fit_one_cycle(3, slice(1e-6,1e-3))
learn_gen.show_results(rows=4)
```

我们可以发现，数字已经被很好的移除了，但是对于Upsampling的部分还是不太好。如果我们想做去水印，我们已经完成了目标。

但是如果我们想让中间的图更像右边的图，该如何做呢？现在我们的Loss function是MSE，实际上，左边图片和右边图片的像素均方差已经非常小。所以这个loss function不能做到我们想要的东西。大多数像素和右边的颜色和接近。但我们丢失了枕头上的纹理、基本全部丢失了眼球、丢失了身体上的纹理。像素均方差认为这是很好的图片了。

![image-20200228130337467](https://tva1.sinaimg.cn/large/00831rSTgy1gccswscgtuj30jn0f5wou.jpg)

### GAN

这个时候，我们为了使用一个更加合适的Loss Function，我们就提到了GAN，GAN的损失函数实际上是调用了另一个模型。来看看下面这个图，我在比对Hi-res Image和prediction的时候不仅仅是用一个Pixel MSE，而是调用了一个binary classification model也就是我们的discriminator去做一个判断，我们给他一些图片，让他判断这是Hi-res图片呢，还是生成图片呢。这就是一个普通的标准二分类交叉熵分类器。我们已经知道怎样做它了。如果有这样一个东西，我们就可以调整generator，不再使用像素均方差做损失度，损失度可以是我们有多擅长瞒过这个critic？我们能不能创建让critic分辨不出的图片？

**使用Pixel MSE原因其实是，GAN会学到很多不在原始图片上的combination，假如我们只用critic来作为loss，那么我们的图片会变得和原图越来越无关，所以这个基础上还要加上Pixel MSE，作为一个新的criteria去限制GAN的创造力。**

所以我们loss function也就是：我们能不能骗过discriminator。好，现在我们要做的就是，用这个critic with 几个批次训练我们的generator，当然最开始的时候是我们的Critic很可怜被我们的generator给骗，所以这样做了一段时间之后呢，我们可以停止训练这个generator，而是去训练这个critic，用新生成的图片，那么现在critic变聪明了，我们回去train generator，然后重复这个过程back and forth。

在fastai里面呢，使用的是新版本的GAN，其实也不能说是新版本，而是预训练版本的GAN，我们的generator和critic都是用的pre-trained model。一般来说，GAN是很难训练的，特别是在一开始的时候，我们的critic和generator都know nothing，训练generator的时候，critic无事可做，轮到critic的时候，generator也无事可做，但是当他们知道自己要干什么的时候，也就是generator知道我是要骗过critic而不是单纯生成图片，critic也知道我不是单纯判断图片，而是要让generator变得更好的时候，实际上训练的过程是会越来越快的。所以一开始是相当于是盲人骑瞎马哈哈哈哈。而使用pre-trained model实际上是越过了这个过程。

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcct16yc0uj313b0ml44u.jpg)

### Critic

刚刚也说了，这个critic也就是一个binary crossentropy model。我们先把generator predict的图片放到新的folder去，以便于我们对critic进行训练。

这是做这个的一小段代码。我们要创建一个叫`image_gen`，把它放进一个叫`path_gen`的变量里。我们写一个叫`save_preds`的小函数，它接收一个data loader。我们取出所有的文件名。对一个item list来说，如果它是一个image item list，`.items`里存的是文件名。这个data loader的dataset里是文件名。现在，我们遍历data loader里的每一个batch，我们取这个batch的预测batch（preds），`reconstruct=True`代表它会为batch里的每一个东西创建fastai图片对象。我们遍历每个预测值，保存它们。我们用和原始文件一样的名字，但是会把它放到新目录里。

```python
name_gen = 'image_gen'
path_gen = path/name_gen
# shutil.rmtree(path_gen)
path_gen.mkdir(exist_ok=True)
def save_preds(dl):
    i=0
    names = dl.dataset.items
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1
save_preds(data_gen.fix_dl)
```

并且生成critic model可以用的data，和以前一样，这是一个从文件夹生成的image item list，这个classes是`image_gen` 和 `images`。我们要做一个随机的分割得到一个验证集，因为我们想知道critic做得怎么样。我们像之前一样按文件夹标注，做一些变形，data bunch，normalize

```python
def get_crit_data(classes, bs, size):
    src = ImageItemList.from_folder(path, include=classes).random_split_by_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data
```

好，现在就可以去训练critic啦。

这里的learner中module用的是`gan_critic()` 而不是ResNet34，原因是Jeremy说，（我没怎么理解这句话）you need to be particularly careful that the generator and the critic can't both push in the same direction and increase the weights out of control. So we have to use something called spectral normalization to make GANs work nowadays，之后part2会讲。

Loss也用了一个没见过的，但是现在可以把它当成CrossEntropyLoss，A GAN critic uses a slightly different way of averaging the different parts of the image when it does the loss, so anytime you're doing a GAN at the moment, you have to wrap your loss function with `AdaptiveLoss`. Again, we'll look at the details in part 2.

metric也是没见过的，但是理解为GAN的accuracy版本。

```python
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)
learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)
```

```python
learn_critic.fit_one_cycle(6, 1e-3)
```

好到此为止，我们也有了一个pre-trained的critic。

### Finishing up GAN

我们已经有了一个pretrain的generator和一个pretrain的critics，

```python
data_crit = get_crit_data(['crappy', 'images'], bs=bs, size=size)
learn_crit = create_critic_learner(data_crit, metrics=None).load('critic-pre2')
learn_gen = create_gen_learner().load('gen-pre2')
```

那我们现在需要的就是在这两个之间来回训练，fastai帮我们创建了一个`GANLearner`，它可以帮我们决定每个model要训练多久，以及学习率是多少。这里的`weights_gen`是因为Pixel MSE和critic这两个loss不在一个scale上，所以我们通常将critic的loss乘上50到200之间的一个数从而使他们compatible。

而且，**GAN不用动量**。当你训练它们时，你在不停的切换generato和rcritic，所以很难用上动量。所以，当你创建Adam优化器时，这个动量的参数（`betas=(0.,...)`），要设为0。

```python
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1.,50.), show_img=False, switcher=switcher, opt_func=partial(optim.Adam, betas=(0.,0.99)), wd=wd)
learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))
```

然后得到的output是like this的，the loss is actually not going down because that's what GAN shows, so you can't tell if the training is good based on the loss value. 这是GAN的一个训练难点。唯一的方式就是看看从每个epoch的结果来看。也就是把上面的`show_img`设为`True`就可以。

![image-20200228155221534](https://tva1.sinaimg.cn/large/00831rSTgy1gccxsc6vyoj30jf0elgp8.jpg)

最后我们来看看结果，第一张图还不错，但是第二张图我们可以看到有一些奇怪的noise，第三张GAN甚至不能把眼珠子填进去。因为它不知道这是一个重要的特征，当然因为我们的critic pretrain mode是pretrain在imagenet上的并且适用于GAN的，那应该是可以找到眼球的，但是这个方法可能会比较麻烦。

![image-20200228155421345](https://tva1.sinaimg.cn/large/00831rSTgy1gccxufnirij30je0j8h0f.jpg)

## Feature Loss

从上面可以看到，眼珠子认不出这个问题比较严重，因为这个feature对于我们来说是很重要的。我们现在用一个类似GAN的方法去让认识到眼珠子是很重要的。我们需要的是一个不一样的损失函数，论文在这里：[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)。也就是Feature Loss。

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcczp71pr7j30s00j8k41.jpg)

这里的fw其实是一个和Unet很相似的网络结构，也可以看做是一个generator。之后呢，他们要做的事情其实就是construct一个和critic很像的loss function但是更加注重Feature。他们当时使用的是VGG16的pretrain model on ImageNet，实际上应该什么pretrained model都可以。

他们把predicted出来的image放到这个VGG16中，最终呢这个模型会告诉你这个图片是个什么，狗还是猫啥的。但是我们其实只想要中间某层的输出，因为recall一下之前那个visualizaiton告诉我们CNN每一层在做什么的时候，随着层数增加我们会发现更多更具体的东西，而在这里，我们需要的就是具体feature的输出，也就是上图每层改颜色的时候（In this case, they've color-coded all the layers with the same grid size and the feature map with the same color. So every time we switch colors, we're switching grid size. So there's a stride 2 conv or in VGG's case they still used to use some maxpooling layers which is a similar idea），激活层的输出。比如这一层的输出是256通道的28x28的matrix，这些28x28的网格大概是用来判断“在这个28x28的网格里，有没有什么毛茸茸的东西？有没有什么发光的东西？有没有什么圆形的东西？有没有什么像眼球的东西？”，然后呢我们可以把真实的y值放到同样的网络里面，取同一个部分的输出，然后比较看match的好不好。这应该能解决我们的眼球问题，因为这里，这个特征图会说“这里有眼球（目标图片），但这里没有（生成图片），所以继续努力，做出一个好点的眼球。”这就是它的思路。这就是feature loss，或者Johnson et al.叫的Perceptual loss。

那同样的，我们开始创造我们的数据集，这里没有给图片随机加数字和quality，只是全部转换成了96x96 with quality 60。

```python
import fastai
from fastai.vision import *
from fastai.callbacks import *

from torchvision.models import vgg16_bn
path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'small-96'
path_mr = path/'small-256'
il = ImageItemList.from_folder(path_hr)

def resize_one(fn,i):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 96, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)

# to create smaller images, uncomment the next line when you run this the first time
parallel(resize_one, il.items)
bs,size=32,128
arch = models.resnet34

src = ImageImageList.from_folder(path_lr).random_split_by_pct(0.1, seed=42)
def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data
data = get_data(bs,size)
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))
```

接着呢，我们要去比较predicted y的feature和ground truth的feature，在这里我们用的L1 loss，当然MSE也可以。接着我们来构造这个VGG16 pretrain model，`.feature`是为了让convolutional part，这里才是我们要的部分，包含在vgg16中，并放在cuda上去运行，但是我们只做evaluation，最后将grad更新给关上。

```python
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)
```

然后，我们要找的是Max Pooling部分，也就是grid size chage的时候，而我们正好想要的事前一步的结果，也就是颜色改变前。

```python
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]
```

接着我们来define我们自己的Feature Loss，当我们调用feature loss类时，我们会传入一些预训练模型，模型的名字是m\_feat。这是包含特征的模型，我们想用feature loss处理它。我们取这个网络的所有层，用里面的特征创建loss。

我们要hook所有这些输出，在PyTorch取中间层的方法就是hook它们。self.hook会存我们勾住的输出。

现在，在feature loss的forward里，我们要调用make\_features，传入traget（实际的y），它会调用VGG模型，遍历所有存储的激活值，取出它们的值。我们对目标out\_feat和输入（generator的输出，in\_feat）都做同样的操作。现在，我们来计算像素的L1损失度。我们遍历所有层的特征，得到它们的L1损失度。我们遍历每个block的最后一层，取出激活值，算出L1。

最后放到叫feat\_losses的list里，把它们加起来。我用list的原因是，我们有这个回调，如果你在损失度函数里把它们放进这个叫.metrics的东西里，它会打印所有层的损失，很方便。

就是这样，这就是我们的perceptual loss或者说feature loss类。

```python
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()
```

然后就可以开始训练啦。这个损失函数使用我们的预训练VGG模型。这个callback\_fns是我提过的LossMetrics，它可以打印出所有层的损失。

```python
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();
```

首先是直接用数据来训练，freeze训练一次，unfreeze训练一次 as usual。

```text
lr = 1e-3
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)

do_fit('1a', slice(lr*10))
learn.unfreeze()
do_fit('1b', slice(1e-5,lr))
```

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd1qziqv1j30od092wh8.jpg)

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd1r4zgqqj30od092act.jpg)

之后我们更换data为size大一点的数据。batch size减半。然后freeze，unfreeze。

```python
data = get_data(12,size*2)
learn.data = data
learn.freeze()
gc.collect()
learn.load('1b');
do_fit('2a')

learn.unfreeze()
do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)
```

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd1rdi1n5j30oh0927bk.jpg)

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd1rj2205j30oh092dn4.jpg)

可以看到效果还不错了。最主要是训练的很快。

### Medium Resolution

当然你还可以尝试用分辨率高一点的图片。

```python
data_mr = (ImageImageList.from_folder(path_mr).random_split_by_pct(0.1, seed=42)
          .label_from_func(lambda x: path_hr/x.name)
          .transform(get_transforms(), size=(1280,1600), tfm_y=True)
          .databunch(bs=1).normalize(imagenet_stats, do_y=True))
data_mr.c = 3
learn.data = data_mr
fn = data_mr.valid_ds.x.items[0]; 

img = open_image(fn); img.shape
torch.Size([3, 256, 320])
p,img_hr,b = learn.predict(img)
show_image(img, figsize=(18,15), interpolation='nearest');
```

原图是这样的：

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd1v72jqhj30t00n941o.jpg)

predict是这样的：

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd1v2otlij30tj0nanm1.jpg)

## RNN

来看看这个图，

* **矩形**代表一个输入，它的形状是批次数量（batch\_size） x 输入数量。
* **箭头**代表一个层，比如说一个矩阵乘积然后一个ReLU。
* 一个**圆圈**是一个激活值。这里，我们有一组隐藏的激活值。这个（第一个箭头）是一个矩阵，尺寸是输入的数量x激活值的数量。输出的尺寸是批次数量x激活值数量。然后，这里有另一个箭头，这代表它是另外一个层，矩阵乘积然后做一个非线性计算。这里，下一层是输出层，所以我们用softmax。
* **三角形**代表一个输出。这个矩阵形状是激活值数量x类别数量，所以我们的输出的尺寸是批次数量x类别数量。

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd242lmyxj30lg0dzmza.jpg)

再来想象一下，我们要把一个很大的文档拆分成3个单词的集合，对每个集合，用前两个单词，预测第三个单词。如果我们现在有这个数据集，我们可以：

1. 把第一个单词作为输入
2. 让它经过一个embedding，创建一些激活值
3. 让它经过一个矩阵乘法和非线性计算
4. 取第二个单词
5. 让它经过一个embedding
6. 然后我们可以把这两个东西加在一起，或者连接在一起（concatenate）。一般来说，你见到两组激活值在一个图里走到一起，你一般可以选择连接或者相加。这会创建第二组激活值。
7. 然后你可以让它经过下一个全连接层和softmax来产生一个输出

这其实是一个完全标准的全连接的神经网络，只不过多了一个小东西，就是在这里做了连接或者相加，我们要用这个网络来通过前两个单词预测第三个单词。

记住，箭头代表层操作，我在这里删除了细节。它们是一个仿射函数和一个非线性计算。

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd28cejyxj30tq0gx42e.jpg)

再来看看这个：

但是我要说的一点是，每次我们从矩形走到圆形，我们在做相同的事情，我们在做embedding。就是一种特别的矩阵乘法，这里你有一个one-hot编码的输入。每次我们从圆形走到圆形，我们取到一个隐藏状态（激活值），通过加上另外一个单词，把它调整成另外一个激活值的集合。然后当我们从圆形走到三角形，我们是在把隐藏状态转成一个输出。我把这些箭头画成不同的颜色。每个箭头应该用相同的权重矩阵，因为它在做相同的事情。

![](https://tva1.sinaimg.cn/large/00831rSTgy1gcd2ahfoyrj30qa0gs426.jpg)

### Human Number

这里验证集是8,000之后的数字，训练集是1到8,000。我们可以把它们放在一起，转成一个data bunch。

```python
src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
data = src.databunch(bs=bs)
```

Check你的数据，这里xxbos是fastai的一个special token，begining of the stream。NLP的analysis来说，数据的开始对于模型训练很重要。

```python
train[0].text[:80]
'xxbos one , two , three , four , five , six , seven , eight , nine , ten , eleve'
```

验证集有13,000个token。就是13,000个单词或者标点符号，所有空格间的东西都是一个token

```python
len(data.valid_ds[0][0].data)
13017
```

这里有个东西叫`bptt`，是指back prop through time which is the sequnece length。对于每一个批次，我们break up成70个words的list来处理。

我们对验证集，取出整个的有13,000个token的字符串，然后我们把它分成64个长度相等的部分。人们经常把这个理解错。我不是说“它们的长度是64”，不是这样。它们是“64个长度基本相等的部分”。我们取这个文档的第一个1/64，然后第二个。

```python
data.bptt, len(data.valid_dl)
(70, 3)

13017/70/bs #bs=64
2.905580357142857
```

我们可以来验证一下，我们取一个data loader的迭代器，取前3个批次（X和Y），我们把元素数量加起来，我们得到的数比13,017少一点，因为最后有一点不够组成一个批次了。

```python
it = iter(data.valid_dl)
x1,y1 = next(it)
x2,y2 = next(it)
x3,y3 = next(it)
it.close()
x1.numel()+x2.numel()+x3.numel()
12928
```

你可以看到，它是95x64的。我声明的是70x64。这是因为我们的语言模型data loader随机改变了`bptt`，来让顺序随机改变（shuffle），有更多的随机性，它对模型有帮助。

```python
x1.shape,y1.shape
(torch.Size([95, 64]), torch.Size([95, 64]))
x2.shape,y2.shape
(torch.Size([69, 64]), torch.Size([69, 64]))
```

好，现在可以来看看language model的x和y是长什么样的。可以看到y是我们的label，是我们要predict的下一个word，也就是x的offset by 1。

```python
x1[:,0]
tensor([ 2, 18, 10, 11,  8, 18, 10, 12,  8, 18, 10, 13,  8, 18, 10, 14,  8, 18,
        10, 15,  8, 18, 10, 16,  8, 18, 10, 17,  8, 18, 10, 18,  8, 18, 10, 19,
         8, 18, 10, 28,  8, 18, 10, 29,  8, 18, 10, 30,  8, 18, 10, 31,  8, 18,
        10, 32,  8, 18, 10, 33,  8, 18, 10, 34,  8, 18, 10, 35,  8, 18, 10, 36,
         8, 18, 10, 37,  8, 18, 10, 20,  8, 18, 10, 20, 11,  8, 18, 10, 20, 12,
         8, 18, 10, 20, 13], device='cuda:0')
y1[:,0]
tensor([18, 10, 11,  8, 18, 10, 12,  8, 18, 10, 13,  8, 18, 10, 14,  8, 18, 10,
        15,  8, 18, 10, 16,  8, 18, 10, 17,  8, 18, 10, 18,  8, 18, 10, 19,  8,
        18, 10, 28,  8, 18, 10, 29,  8, 18, 10, 30,  8, 18, 10, 31,  8, 18, 10,
        32,  8, 18, 10, 33,  8, 18, 10, 34,  8, 18, 10, 35,  8, 18, 10, 36,  8,
        18, 10, 37,  8, 18, 10, 20,  8, 18, 10, 20, 11,  8, 18, 10, 20, 12,  8,
        18, 10, 20, 13,  8], device='cuda:0')
```

然后也可以根据上面这些index，去看vocab里面具体的单词是什么，你在这里可以看到`xxbos eight thousand one`，但在`y`里，没有`xxbos`，只是`eight thousand one`。`xxbos`之后是`eight`，`eight`之后是`thousand`，`thousand`之后是`one`。注意这里是第一个mini批次在8023结束。

```python
v = data.valid_ds.vocab
v.textify(x1[:,0])
'xxbos eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve , eight thousand thirteen , eight thousand fourteen , eight thousand fifteen , eight thousand sixteen , eight thousand seventeen , eight thousand eighteen , eight thousand nineteen , eight thousand twenty , eight thousand twenty one , eight thousand twenty two , eight thousand twenty three'
v.textify(y1[:,0])
'eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve , eight thousand thirteen , eight thousand fourteen , eight thousand fifteen , eight thousand sixteen , eight thousand seventeen , eight thousand eighteen , eight thousand nineteen , eight thousand twenty , eight thousand twenty one , eight thousand twenty two , eight thousand twenty three ,'
```

要查看8023接下来的东西，你要去看第二个mini批次和第三个mini批次的内容。

```python
v.textify(x2[:,0])
', eight thousand twenty four , eight thousand twenty five , eight thousand twenty six , eight thousand twenty seven , eight thousand twenty eight , eight thousand twenty nine , eight thousand thirty , eight thousand thirty one , eight thousand thirty two , eight thousand thirty three , eight thousand thirty four , eight thousand thirty five , eight thousand thirty six , eight thousand thirty seven'
v.textify(x3[:,0])
', eight thousand thirty eight , eight thousand thirty nine , eight thousand forty , eight thousand forty one , eight thousand forty two , eight thousand forty three , eight thousand forty four , eight thousand forty'
```

然后就是大的batch number 2。你会发现他们是接在一起的。

```python
v.textify(x1[:,1])
', eight thousand forty six , eight thousand forty seven , eight thousand forty eight , eight thousand forty nine , eight thousand fifty , eight thousand fifty one , eight thousand fifty two , eight thousand fifty three , eight thousand fifty four , eight thousand fifty five , eight thousand fifty six , eight thousand fifty seven , eight thousand fifty eight , eight thousand fifty nine , eight thousand sixty , eight thousand sixty one , eight thousand sixty two , eight thousand sixty three , eight thousand sixty four , eight'
v.textify(x2[:,1])
'thousand sixty five , eight thousand sixty six , eight thousand sixty seven , eight thousand sixty eight , eight thousand sixty nine , eight thousand seventy , eight thousand seventy one , eight thousand seventy two , eight thousand seventy three , eight thousand seventy four , eight thousand seventy five , eight thousand seventy six , eight thousand seventy seven , eight thousand seventy eight , eight'
v.textify(x3[:,1])
'thousand seventy nine , eight thousand eighty , eight thousand eighty one , eight thousand eighty two , eight thousand eighty three , eight thousand eighty four , eight thousand eighty five , eight thousand eighty six ,'
v.textify(x3[:,-1])
'ninety , nine thousand nine hundred ninety one , nine thousand nine hundred ninety two , nine thousand nine hundred ninety three , nine thousand nine hundred ninety four , nine thousand nine hundred ninety five , nine'
```

数据看完了我们来看看模型。它包含一个embedding（绿色箭头），一个隐藏状态到隐藏状态，棕色箭头的层，一个隐藏状态到输出。每个标颜色的箭头都有一个矩阵。在forward里，我们取第一个输入`x[0]`，让它通过输入到隐藏状态层（绿色箭头），创建我们第一个激活值集合，我们叫它`h`。假设有第二个单词，因为有时我们可能是在批次的最后，这就没有第二个单词。假设有第二个单词，我们把`h`加到`x[1]`的结果上，让它通过绿色箭头（`i_h`）。然后我们可以说，好了，我们的新`h`是这两个相加的结果，放入隐藏状态到隐藏状态（棕色箭头），然后ReLU，然后batch norm。然后，对这第二个单词，做相同的事情。最后经过蓝色箭头，放入`h_o`。

![image-20200315221150600](https://tva1.sinaimg.cn/large/00831rSTgy1gcvoxw0n4fj31gy0tqkaa.jpg)

上面有很多replicate的code，我们想要refactor让他们变得更加简洁。好了，我们把它重构进了一个循环。我们要在循环里对每一个`x`里的`xi`，做这个。这就是RNN。RNN就是一个重构。没有什么新东西。现在，它是一个RNN了。

```python
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)  # green arrow
        self.h_h = nn.Linear(nh,nh)     # brown arrow
        self.h_o = nn.Linear(nh,nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[1], nh).to(device=x.device)
        for xi in x:
            h += self.i_h(xi)
            h  = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)
```

我们来重构我们的图，使用循环的一个好处是，如果我不是用前三个单词预测第四个，而是用前八个预测第九个。它也可以做到。可以用它来处理任意长的序列。

![img](https://tva1.sinaimg.cn/large/00831rSTgy1gcvp4h25joj30ty0gnn1n.jpg)

我们来把`bptt`变成20。现在，我们尝试不再从前n-1个单词预测第n个单词，比如用第一个单词预测第二个、用第二个单词预测第三个、用第三个单词预测第四个等等。先看看我们的损失函数，我们之前是用target的最后一个来算loss，但是这很费时，因为序列里有很多单词。

![img](https://tva1.sinaimg.cn/large/00831rSTgy1gcvp6d2ulej30ft01jaab.jpg)

我们可以尝试来把`x`的每个单词和`y`里的每个单词做比较，要做到这个，我们要改变这个图，不再是只在循环的最后有一个三角，而是三角进到训练里边，换句话说，在每个循环后，预测，循环，预测，循环，预测。

![img](https://tva1.sinaimg.cn/large/00831rSTgy1gcvp7ey3faj30tk0gogmj.jpg)

而code就变成了这样，它和以前的一样，只是现在我创建了一个数组，每次在循环时，把`h_o(h)`加入到这个数组里。现在对于n个输入，我创建了n个输出。每个单词后都做预测。

```python
data = src.databunch(bs=bs, bptt=20)
x,y = data.one_batch()
x.shape,y.shape
(torch.Size([45, 64]), torch.Size([45, 64]))
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[1], nh).to(device=x.device)
        res = []
        for xi in x:
            h += self.i_h(xi)
            h = self.bn(F.relu(self.h_h(h)))
            res.append(self.h_o(h))
        return torch.stack(res)
```

但是这个时候我们却发现，accuracy变差和之前把三角形放进来的时候的相比较。这是因为我在预测第二个单词时，只有一个单词的状态可以用。当我预测第三个单词时，只有两个单词的状态可以用。这是一个更难解决的问题，那怎么解决这个accuracy变差的问题呢，可以看到有一行是加上一个内容为0的matrix，那我们可以考虑将他变成不在是每一次都做另外一个bptt时，都重置，我们可以carry on，因为本身我们的batch也是有顺序并且相接的。

```python
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(x.shape[1], nh).cuda()

    def forward(self, x):
        res = []
        h = self.h
        for xi in x:
            h = h + self.i_h(xi)
            h = F.relu(self.h_h(h))
            res.append(h)
        self.h = h.detach()
        res = torch.stack(res)
        res = self.h_o(self.bn(res))
        return res
```

就是这样，现在它是`self.h`。这还是相同的代码，但是最后，我们把新的`h`赋值给`self.h`。它现在在做相同的事，但是不会丢掉这个状态。现在，我们的结果比上次好了。准确率上升到了54%。这就是RNN。你要一直保存状态。要记住，RNN没什么特别的，它就是一个普通的全连接网络，只是重构了一个循环。

接着呢，我们也可以再提高accuracy，通过在每个循环最后，你可以不单单输出一个结果，你可以把它输出到另外一个RNN里。你可以从一个RNN进到另外一个RNN。

![img](https://tva1.sinaimg.cn/large/00831rSTgy1gcvpdo8u04j31400m2dhb.jpg)

要做到这个，我们要再重构。我们复制`Model3`的代码，用PyTorch内置的代码替换它，可以这样写。

`nn.RNN`就是说为我做循环。我们还是会用相同的embedding、相同的输出、相同的batch norm、相同的`h`初始化，但是没有了循环。RNN的好处是你现在就可以说你想要多少层。

```python
class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.rnn = nn.RNN(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(1, x.shape[1], nh).cuda()

    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))
```

所以我们的有和没有loop的图像对比是这样的：

![img](https://tva1.sinaimg.cn/large/00831rSTgy1gcvpjwxakyj314p0mgqb7.jpg) ![img](https://github.com/hiromis/notes/raw/master/lesson7/55.png)

我们用了20的BPTT，所以这里有20层。我们从可视化损失空间的论文里知道，深度网络有很曲折的损失平面。当我们创建很长的很多层的网络时，它很难训练。那怎么办呢，有一些可以使用的方法。

一个是你可以添加skip connection，人们经常不再单单把这些加在一起（绿色箭头和橙色箭头），它们用一个小的神经网络决定要保持多少绿色的箭头，多少橙色的箭头。当你这样做时，你得到了叫GRU或LSTM的东西。现在，我们可以用一个GRU（也就是上面的init中的一个函数）来替代。它和我们之前的很像，只是它可以在更深的网络里处理更长的序列。我们用两层。然后！accuracy就提升到了75%。这就是RNN！

这样，你有一个有n个输入的序列，一个有n个输出的序列，我们用它来做语言模型，你可以把它用在其它任务上。

比如，输出的序列可以用在每个单词上，来判断有没有什么东西很敏感，是不是要消除它。可以判断是不是私有的数据。它可以用来为单词做语音标注。可以来判断是不是要格式化单词。这些都被叫做**序列标注任务（sequence labeling task）**，你可以用相同的方法来做所有的序列标注任务。你可以像我在前面课里做的一样，在做完语言模型后，去掉`h_o`，放入一个标准的分类头，然后，你可以做NLP分类了，这会给出一个非常好的结果，即使是很长的文档。这是超级有用的技术，不是遥远神秘的魔法。

