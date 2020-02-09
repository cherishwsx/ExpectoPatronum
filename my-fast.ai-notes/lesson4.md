---
description: >-
  Original Detailed Notes:
  https://github.com/hiromis/notes/blob/master/Lesson4.md
---

# Lesson 4

## Interesting Question

### Question 1

> How to combine NLP \(tokenized\) data with meta data \(tabular data\) with Fastai? For instance, for IMBb classification, how to use information like who the actors are, year made, genre, etc

In the neural network, you can have two different sets of inputs merging together into some layer. It could go into an early layer or into a later layer, it kind of depends. If it's like text and an image and some metadata, you probably want the **text going into an RNN, the image going into a CNN, the metadata going into some kind of tabular model like this.** And then you'd have them basically all **concatenated together**, and then go through some fully connected layers and train them end to end.

### Question 2

> Language model that deals with emoji or Hindi word?

Pretrained model 可能不会有大量的emoji在backend data中，但是可以通过微调。但是比如对于中文来说，it's better to use a model that is specifically trained with Chinese character. And there is a repo called [Model Zoo](https://forums.fast.ai/t/language-model-zoo-gorilla/14623) for fast.ai where there are a lot of pretrained model in different area like gene sequences, medical text and musix note.

### Question 3

> The other thing you can do if you, for whatever reason, can't go through that UX of asking people did you like those things \(for example if you're selling products and you don't really want to show them a big selection of your products and say did you like this because you just want them to buy\), what you can do?

## Tabular Data

Traditionally, people use regression, random forest and machine learning to work with tabular data. But with deep learning, feature engineering becomes pretty manageble. Pinterest replaced their gradient boosting machines for recommendation to deep learning. Hand creative features is not necessary anymore so maintenance is less. But before fastai.tabular, there isn't any library that could develop a easy neural network way to work with tabular data.

### Create the DataBunch

1. Specify the name for categorical variables and continuous variables.
   1. For categorical, we do **embedding**.
   2. For continuous, we just treat them like pixels value.
2. 对比Feature Engineer和Data Augmentation
   1. Feature Engineer是提前对数据做的一种统一的处理方式，不是边训练边处理的。

      ```python
      procs = [FillMissing, Categorify, Normalize]
      # fastai has many built-in ways of preprocessing.
      # 这里的FillMissing用中位数，并增加一列column去indicate是否缺失
      ```

   2. Data Augmentation是随机的对图片进行的处理。
3. Train and Test split 通常来说，表格数据会有时间序列，或者地理数据。data的顺序很重要，所以在split的过程中，选择某一段连续的data作为test从而保证不会cheating。同理对于视频帧数的选择也是如此。
4. `tabular_learner` for tabular model，其中layers这个参数就和ResNet50的含义是一样的。

   ```python
   learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
   ```

## Collaborative filtering

* Most basic collab filtering is that a table user ID and the movieID they bought or like. Then you can add more information like comments, reviews, rating and transaction time etc. Two ways of showing this dataset, 回忆一下之前AI的课程，在介绍协同过滤的时候，table所呈现的样子基本都是所有的user是row，所有的item是column，这样每个用户对于每个item的打分也更加清晰，但是因为不是每个用户都会打分所有的item，所以在一定程度上这个matrix是sparse的，从而也导致了不容易存储，也非常巨大 ，所以在存储的时候还是按照long format的形式去存储
* 但是实际上我们之后的alternative least square就是在学习填满这个matrix，**Matrix Factorization** 将两个randomly initialize的matrices做dot product得到一个用户对于一个电影的评分，然后和真实情况比较，然后通过loss function计算loss，之后Gradient Descent update参数 to make the loss smaller。[1:07:25](https://youtu.be/C9UdVPE3ynA?t=4045)

* 适用的情况下是，当你已经有users or movies, product's information.
* 其实在fastai中，collab filtering只是一个简单的线性神经层，也就是`nn.Embedding`这一层，with two weight matrix with a nonlinearity.
* 我们需要去设定rating的范围吗，也就是score的定义，我们可以不用，因为神经网络自己会学习到真实的范围，但是为了让神经网络focus on重要的tasks我们就在forward最后输出的地方加上了一个sigmoid。

![img](https://github.com/hiromis/notes/raw/master/lesson4/6.png)

### Read Data

```python
data = CollabDataBunch.from_df(ratings, seed=42)
```

### Learner

```python
y_range = [0,5.5] #score
learn = collab_learner(data, n_factors=50, y_range=y_range)
```

n\_factors is the latent matrix rank.

### Cold Start Problem

#### User End

当一个推荐系统面对一个全新的用户时，系统中并没有关于这个新用户的任何数据，所以很难在一开始就给他们推荐他们想要的电影。解决方法有：

1. 推荐热门物品或者必需品，这些物品往往是热点或者是购买最多的，比如视频推荐的新上映的大片，电商的畅销品，或者是生活必需品。
2. 基于用户的信息来做推荐，如年龄，性别，地域等。这要求平台事先要知道用户的部分信息，这在某些行业是比较困难的，比如OTT端的视频推荐。
3. 将库中的物品聚类，在给新用户推荐时，每个类别中推荐几个，总有一款是你喜欢的。
4. 当用户有很少的行为记录时，这时很多算法（比如协同过滤）还无法给用户做推荐，这时可以采用基于内容的推荐算法。
5. 当产品在拓展过程中，比如视频类应用，前期只做长视频推荐，后来拓展到短视频，那么对某些没有短视频观看行为的用户，怎么给他做短视频推荐呢？可以行的方式是借用迁移学习的思路，利用长视频观看历史，计算出用户的相似度，如果某个用户没有看短视频，但是跟他相似的用户看了很多短视频，这时可以将相似用户看过的短视频推荐给该用户
6. 事先构造选项，让用户选择自己的兴趣。What Neflix used to do is use UX to fix this problem so that avoid cold start problem.
7. 利用社交信息来做冷启动，特别是在有社交属性的产品中，将好友喜欢的物品推荐给新用户。

   来源：[知乎](https://www.zhihu.com/question/19843390/answer/343050630)

#### Product End

1. 基于物品的属性的推荐，一般新上线的物品或多或少都是有一些属性的，根据这些属性找到与该物品最相似的物品，

   这些相似的物品被哪些用户“消费”过，可以将该物品推荐给这些消费过的用户。

2. 另外一种思路是借用强化学习中的exploration-exploitation思想，将该物品曝光给随机一组用户，观察用户对物品的反馈，找到对该物品有正向反馈\(观看，购买，收藏，分享等\)的用户, 后续将该物品推荐给与正向反馈相似的用户。It's not really a problem because like the first hundred users who haven't seen the movie go in and say whether they liked it, and then the next hundred thousand, the next million, it's not a cold start problem anymore.

#### Model End

The other thing you can do if you, for whatever reason, can't go through that UX of asking people did you like those things \(for example if you're selling products and you don't really want to show them a big selection of your products and say did you like this because you just want them to buy\), what you can do?

Meta data driven model which is different than collaborative filtering can use geographical location, gender and age to do initial recommendation guess.

### Code Detail

#### `get_collab_learner`

Take a look at the `get_collab_learner`

![11.png](https://github.com/hiromis/notes/blob/master/lesson4/11.png?raw=true)

The models that are being created for you by fastai are actually PyTorch models. And a PyTorch model is called an `nn.Module` that's the name in PyTorch of their models. It's a little more nuanced than that, but that's a good starting point for now. When a PyTorch `nn.Module` is run \(when you calculate the result of that layer, neural net, etc\), specifically, it always calls a method for you called `forward`. So it's in here that you get to find out how this thing is actually calculated.

When the model is built at the start, it calls this thing called `__init__` as we've briefly mentioned before in Python people tend to call this "dunder init". So dunder init is how we create the model, and forward is how we run the model.

One thing if you're watching carefully, you might notice is there's nothing here saying how to calculate the gradients of the model, and that's because PyTorch does it for us. So you only have to tell it how to calculate the output of your model, and PyTorch will go ahead and calculate the gradients for you.

So in this case, the model contains:

* a set of weights for a user \(embedding\)
* a set of weights for an item \(embedding\)

forward中直接就点乘了embedding然后加上bias。

* a set of biases for a user
* a set of biases for an item

And each one of those is coming from this thing called `embedding`. Here is the definition of `embedding`:

[![img](https://github.com/hiromis/notes/raw/master/lesson4/12.png)](https://github.com/hiromis/notes/blob/master/lesson4/12.png)

All it does is it calls this PyTorch thing called `nn.Embedding`. In PyTorch, they have a lot of standard neural network layers set up for you. So it creates an embedding. And then this thing here \(`trunc_normal_`\) is it just randomizes it. This is something which creates normal random numbers for the embedding.

#### Embedding

An embedding, not surprisingly, is a matrix of weights. Specifically, an embedding is a matrix of weights that looks something like this:

[![img](https://github.com/hiromis/notes/raw/master/lesson4/13.png)](https://github.com/hiromis/notes/blob/master/lesson4/13.png)

In our case, we have an embedding matrix for a user and an embedding matrix for a movie.

#### Bias

* Bias is the term that you want to add to the matrix multiplication to indicate the preference of users or popularity of the movies.
* It's pretty much like the intercept. Remember how I said there's this idea of bias and the way we dealt with that in our gradient descent notebook was we added a column of 1's. So we don't just want to have prediction equals dot product of these two things, we want to say it's the dot product of those two things plus a bias term for a movie plus a bias term for user ID.

#### Overview of important terminology

* Input

You've got lots of pixels, but let's take a single pixel. So you've got a red a green and a blue pixel. Each one of those is some number between 0 and 255, or we normalize them so they have the mean of zero and standard deviation of one. But let's just do 0 to 255 version. So red: 10, green: 20, blue 30.

So what do we do with these? Well, what we do is we basically treat that as a vector, and we multiply it by a matrix. So this matrix \(depending on how you think of the rows and the columns\), let's treat the matrix is having three rows and then how many columns, which is the parameter size. Initially, the matrix that you multiply is initialized by random. Then you multiply them and get through a activation matrix. And the general process is weight matrix→ReLU→ weight matrix→ReLU→ weight matrix→ final output. s

![img](https://github.com/hiromis/notes/raw/master/lesson4/17.png)

* Weights/parameters
  * Random
* Activations
* Activation functions / nonlinearities
* Output

![img](https://github.com/hiromis/notes/raw/master/lesson4/18.png)

* Loss

Compare the final output decision with the real case to calculate the loss function and go back to update the parameter.

* Metric
* Cross-entropy
* Softmax
* Fine tuning
  * Layer deletion and random weights
  * Freezing & unfreezing

