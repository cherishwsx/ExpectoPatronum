---
description: >-
  Original Detailed Notes:
  https://github.com/hiromis/notes/blob/master/Lesson3.md
---

# Lesson 3

## General Issue&Tips

### `lr vs. slice(lr)`

With the former, every parameter group will use a learning rate of lr, whereas with the latter, the last parameter group will use a learning rate of lr, while the other groups will have lr/10. say you have 3 layer groups: group 1, 2 and 3. max\_lr=slice\(1\) means that the learning rate for group 3 is 1, and 0.1 for groups 1 and 2. max\_lr=1 means the learning rate is 1 for groups 1, 2 and 3.

### Different learning rate shape before and after unfreezing

Before unfreezing the model,

![image-20200207163732097](https://tva1.sinaimg.cn/large/0082zybpgy1gbop2v5d4nj30ku0csgmt.jpg)

Then you can easily find the steepest slope range.

After unfreezing the model,

![image-20200207163803837](https://tva1.sinaimg.cn/large/0082zybpgy1gbop3f9scmj30l20c60tq.jpg)

More likely, you will get this kind of shape, find the point that just before to increase rapidly and then go back 10 times to start. And stops at the best learning rate we found before unfreeze the model and devided by 5 or 10. \(discrimitive learning rate\)

```python
learn.fit_one_cycle(5, slice(1e-5, lr/5))
```

### Image Size for Training

Sometimes the pretrain model we get is only good for certain size of a image and we usually resize the image. But resizing always may lead to increasing loss so instead we can use higher resolution or even original image size. But how can we get a model that is good for this size? Say, We can fine tune a model that is good for 128x128 to make it good at 256x256. **这个就是为什么不直接使用ImageDataBunch来读数据**，而是先用DataLoader再转换成DataButch，这样我们可以多次create dataset。

1. Replace with your new data in the learner

   \`\`\`python

   learn = cnn\_learner\(data\_128, arch, metrics=\[acc\_02, f\_score\]\)

data\_256 = \(src.transform\(tfms, size=256\) .databunch\(\).normalize\(imagenet\_stats\)\)

learn.data = data\_256 \#将模型中的data替换成新的data

```text
2. freeze it if your learner is unfreezed before
3. Find the learning rate and plot it
4. Then can also unfreeze it to train the model with previous 128 unfreeze learning rate for training. Or create a new learner with bigger size dataset and load the weight from learner with smaller dataset to retrain it.

More detail on **Progressive Resizing**. Here we mainly benefit the result that we got when using smaller size picture and don't need to spend too much time find the optimal for bigger size image from scratch.

## Save model Info
* stage-1 refers to freezed model
* stage-2 refers to unfreezed model
* image size
* model archetecture

## Check Memory Used
```python
# Below are equivalent
gpu_mem_get_free_no_cache()
torch.cuda.empty_cache()
```

### Video Dataset Validation

Normally, when splitting the frames from video dataset, we can't do randomly split because it's possible that two continuous frame one goes to the training set and the other goes to the validation and this is cheating.

### User Defined Metric

Very useful when you are doing Kaggle competition since they can ask for different metric than built-in metric function.

### About Fit-One-Cycle

When looking at the training loss and validation loss at fast.ai

```python
learn.recorder.plot_losses()
```

![image-20200207163926775](https://tva1.sinaimg.cn/large/0082zybpgy1gbop4uvjmtj30km0cy403.jpg)

It goes up and down. Because that's our fit-one-cycle does, it starts at the bottom and goes up and goes down again.

```python
learn.recorder.plot_lr()
```

![image-20200207163959461](https://tva1.sinaimg.cn/large/0082zybpgy1gbop5fvpsfj30kk0c6q4a.jpg)

* Normally, when the loss is really close to the optimal, **you want your learning rate to be small at the end** called **learning Rate Annealing**, since you don't want your loss jump back and forth at the boundary and never reach the lowest point.

  ![enter image description here](https://github.com/hiromis/notes/raw/master/lesson3/whiteboard.gif)

  ​

* But **learning rate small at the beginning** is kind of new concept. \(Leslie Smith\). With the learning rate small at the beginning will possibly lead you to local min that will not get you the model with good solution in all situation, so you want to get to the flat area. Then you need a bigger learning rate to jump out of it to explore the whole function surface.

![image-20200207164040521](https://tva1.sinaimg.cn/large/0082zybpgy1gbop64u8o0j30ku0ag0v8.jpg)

#### Idea case

This is actually a really good situation that you want since you started to get decreasing loss after getting a increasing meaning that you find a really good learning rate. So if you unfreeze your model and plot the losses and found that your loss is always decreasing little by little you can try pass a bigger learning rate to kind of see an increase in this kind of shape. And if your accuracy is still improving you can try keep training it and when it starts getting worse then that means it is overfit.

![image-20200207164057033](https://tva1.sinaimg.cn/large/0082zybpgy1gbop6eusl0j30ke0d0taa.jpg)

## Activation Function

 

![Alt text](https://tva1.sinaimg.cn/large/0082zybpgy1gbop8glb3aj30rx0lrgoh.jpg)

1. After we are done with the convolution \(maybe to next convolution or top layer result\), we will apply a non-lineariy function or we call it activation function. It takes the result from the convolution and then put it into another function.

* Sigmoid is less commonly used
* ReLU is more commonly used. `max(x, 0)`
* **Universal approximation theorem** which means that with different number combination of convolution \(matrix multiplication\) and activation function you can appriximate any function. So now what you left is to find out the parameter matrix in your matrix multiplication using Gradient Descent to do the work you want.
* 直白的解释：We have a function where we take our input pixels or whatever, we multiply them by some weight matrix, we replace the negatives with zeros, we multiply it by another weight matrix, replace the negative zeros, we do that a few times. We see how close it is to our target and then we use gradient descent to update our weight matrices using the derivatives, and we do that a few times. And eventually, we end up with something that can classify movie reviews or can recognize pictures of ragdoll cats. That's actually it.

## Kaggle API

1. Run below to install kaggle tool

   ```python
   ! {sys.executable} -m pip install kaggle --upgrade
   ```

2. Go to your account and scroll down to find your credencial which is a jason file. Download that upload it to your machine.
3. Run below to get your credential to .kaggle path.

   ```python
   ! mkdir -p ~/.kaggle/
   ! mv kaggle.json ~/.kaggle/
   find . -name .kaggle # if you don't know where it is
   ```

4. Then downloard the file from competetion which **you have to accept the rule otherwise you will get 403 error**.  See [here](https://medium.com/@ankushchoubey/how-to-download-dataset-from-kaggle-7f700d7f9198) for more info on Kaggle API.

   ```python
   ! kaggle competitions download -c planet-understanding-the-amazon-from-space
   ```

   在这个例子下，notebook提供的命令行都出现了404，解决办法是去网页上手动下载然后upload到instance，然后再move to指定的位置进行unzip

5. Install the 7z unzip. Note we can use --yes to answer the conda question.

   ```python
   ! conda install --yes --prefix {sys.prefix} -c haasad eidl7zip
   ```

## Data Block API

Built-in Fast.ai function that helps you get different form of data into your model. And this is related to a topic that I mentioned in the last lecture ImageList which is exactly one of a data block API. Think of it as a step before DataBunch. More detail [here](https://github.com/fastai/fastai/blob/master/docs_src/data_block.ipynb)

1. `from_folder`

   ```python
   data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_folder()              #How to split in train/valid? -> use the folders
        .label_from_folder()            #How to label? -> depending on the folder of the filenames
        .add_test_folder()              #Optionally add a test set (here default name is test)
        .transform(tfms, size=64)       #Data augmentation? -> use tfms with a size of 64
        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch
   ```

2. `from_folder` with user-defined function

```python
data = (SegmentationItemList.from_folder(path_img)
        #Where to find the data? -> in path_img and its subfolders
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_func(get_y_fn, classes=codes)
        #How to label? -> use the label function on the file name of the data
        .transform(get_transforms(), tfm_y=True, size=128)
        #Data augmentation? -> use tfms with a size of 128, also transform the label images
        .databunch())
        #Finally -> use the defaults for conversion to databunch
```

1. `from_csv`

```python
data = (ImageList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        #Where to find the data? -> in planet 'train' folder
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(label_delim=' ')
        #How to label? -> use the second column of the csv file and split the tags by ' '
        .transform(planet_tfms, size=128)
        #Data augmentation? -> use tfms with a size of 128
        .databunch())                          
        #Finally -> use the defaults for conversion to databunch
```

### `class Dataset`

![Alt text](https://tva1.sinaimg.cn/large/0082zybpgy1gbop9d7cqlj31020cmqbt.jpg)

Actually does nothing, it only return the length of the data and return a element by using a index. But clearly this is not enough for giving the model since we need a mini-batch for GPU to do computation parrallely. So here we have `DataLoader`

### `DataLoader`

![Alt text](https://tva1.sinaimg.cn/large/0082zybpgy1gbop9oif1zj310c0e4wmw.jpg)

Takes the dataset and create the mini-batch which you determine the size. 这一步相当于是让机器知道你要分配多少资源去做接下来的计算，但是仍然不够去验证模型的好坏，这时候就有`DataButch`

### `DataBunch`

![Alt text](https://tva1.sinaimg.cn/large/0082zybpgy1gbop9z0penj31020a0wjp.jpg)

这里面可以看到你take train dataloader and val dataloader from previous step to feed into the model. 下面这个例子就是把data变成DataLoader然后是DataButch的过程。

```python
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' ')) # this label_delim works for multiple labels for one image. And check how your csv file separate the multiple labels.
data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))
```

* `.datasets()` actually turns them into PyTorch datasets
* `.transform()` is the thing that transforms them
* `.databunch()` is actually going to create the the DataLoader and the DataBunch in one go

## get\_transforms

* `max_warp` 即变形，改变对图片的查看方向，比如从上看，从下看和从左右看，图片的形状是会改变的。fast.ai是第一个提供快速运行warping的library。
* `do_flip`包括左右颠倒，默认为`True`.
* `flip_vert` 包括8个对称翻转，比如90度翻转。默认为`False`对于没有垂直方向定义的图片可以设置为True, 比如医疗影像，地理影像等。但是对于猫狗图片，最好是设置水平翻转。

### Multi Label Metric

* 当你一个图片对应多个label时，模型output出的对应`data.c`那么多个的prob，不能再用`argmax`去得到最后的结果。这个时候会借助`threshhold`的帮助，让每个类别的概率和阙值相比较。
  * `accuracy_thresh` which can imcorporate the thresh into traditional accuracy that uses `argmax`. This will return all the label that bigger than the threshold.

    ```python
    def acc_02(inp, targ):
    return accuracy_thresh(inp, targ, thresh=0.2)
    ```

    which is the same with below that supported by python3.

    ```python
    partial(accuracy_thresh, thresh=0.2)
    ```
* `label_from_df(label_delim=' ')` when you need to identify multiple lables.

## Image Segmentation

### What is Image Segmentation?

We want to change this

![Alt text](https://tva1.sinaimg.cn/large/0082zybpgy1gbopaipz9mj30y80oqhb9.jpg)

to this

![Alt text](https://tva1.sinaimg.cn/large/0082zybpgy1gbopangrvlj30x40oyadc.jpg)

Basically, we want to label each object we segmented. And it's a classification problem if we break down for one object in the image, say for the top right corner pixels, is it a tree, pet or person? And if we extend the classification to every pixels then it's segmentation.

### Preparation

Usually, you need a dataset that has already labeled or **segment masks**\(which is a interger array instead of regular image floating array\), that is, each pixels has their color label. So it's hard to do it yourself.

### Check Image and Mask

* For regular image with floating array:

  ```python
  img = open.image(file_path)
  img.show()
  ```

* For mask with interger array:

  ```python
  mask = open.mask(file_path)
  mask.show()
  ```

### Codes for Mask

Usually there is a file indicating the object that each integer number represents in order.

![image-20200207164542340](https://tva1.sinaimg.cn/large/0082zybpgy1gbopbdspdoj30u00u3n47.jpg)

Top left is 21 and let's see what is 21.

![image-20200207164554731](https://tva1.sinaimg.cn/large/0082zybpgy1gbopbkee8qj310a0e8n1p.jpg)

21 is sky!

### Transformation Problem for Mask

So here for segmentation problem, we basically dealing with a image label and so when we flipping the x, we want to flip the y as well in order to match them.

### Segmentation`show_batch`

When calling `show_batch`, fast.ai will know you are working on segmentation since previous steps, so it will combine the mask and original image together for you. ![image-20200207164606107](https://tva1.sinaimg.cn/large/0082zybpgy1gboqy3zkuhj30kk0g0jsh.jpg)

### Archetecture

Use **U-net** for segmentation.

![image-20200207164623373](https://tva1.sinaimg.cn/large/0082zybpgy1gbopc21mioj310o0os433.jpg)

```python
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
```

## Regression Model

### Dateset Illustration

1. For this head pose dataset, we want the model to give us two number which is a coordinate to locate the center of the face. So here the label would be the image with a dot on the center of their face.
2. Similarly, these images from videos, we choose the validation to be totally different person from the train dataset.
3. And when we do transformation, the center of their face is also moving, so we want also transform the y when we transform the x.

### Loss Metric

```python
learn = cnn_learner(data, models.resnet34)
learn.loss_func = MSELossFlat()
```

## NLP

### Basic Procedure

* Using fastai.text application and the basic step is
* Read the dataset
* Tokenization

  ![image-20200207164646687](https://tva1.sinaimg.cn/large/0082zybpgy1gbopch4et5j31000q07er.jpg)

  The texts are truncated at 100 tokens for more readability. We can see that it did more than just split on space and punctuation symbols:

* the "'s" are grouped together in one token
* the contractions are separated like this: "did", "n't"
* content has been cleaned for any HTML symbol and lower cased
* there are several special tokens \(all those that begin by xx\), to replace unknown tokens \(see below\) or to introduce different text fields \(here we only have one\).`xxfld` is when you have title or summary indicating the position of the passage. `xxcap` means you have a word that is all capitalized. And other word with `xx` will be unknown token that is not common enough to be in the vocabulary.
* Numericalization Token之后我们会有一个vocabulary，我们想将vocabulary变成一个矩阵，每一个词对应一个parameter weight，为了防止这个矩阵过大， 我们by default限制vocabulary的大小不超过60,000。频率低于2的词语会被从Vocab中舍弃。 We first have to convert them to numbers. This is done in two differents steps: tokenization and numericalization. A TextDataBunch does all of that behind the scenes for you.

  ```python
  TextDataBunch.from_csv(path, 'texts.csv')
  ```

### Model

In our case, we analyze the movie reviews and do _sentimental_ analysis. And we are gonna use transfer model with a pretrained model called language model that are not limited to movie reviews. 我们现在拥有的数据是25,000条review和对应的label，如果从scratch with random parameter initialization来训练模型，很显然这是不足够的。Basically, how to speak English﹣how to speak English well enough to recognize they liked this or they didn't like this. Sometimes that can be pretty nuanced. Particularly with movie reviews because these are like online movie reviews on IMDB, people can often use sarcasm. It could be really quite tricky.

#### Language Model

* A **language model** is a model that learns to predict the next word of a sentence. To predict the next word of a sentence, you actually have to know quite a lot about English \(assuming you're doing it in English\) and quite a lot of **world knowledge**. 学习的不仅仅是语言，更多的是知识。
* 从High Level上来理解，神经网络运用了，say movie review has 2000 words，这 2000个词语去学习如何去predict，从信息量上而言，我们不仅仅是运用词频（和下面N-grams对比），而是学习了每一个word。再加上有transfer learning，我们相当于有了一个庞大的语料库的信息。
* Wikitext 103 is the backend dataset. Billion tokens to predict and every time we make mistake, we calculate our loss and parameters and update the models.
* 这个pretrain model and later fine tuning是为了让Model学习电影评论的词语预测，这个过程没有外界人工的label也可以进行，called **self supervised learning**
* 手机里的输入法实际上就是运用了一个mini language model去预测你的下一个输入的词。
* 在fine tune时，我们要使用的是所有text data regardless of validation and train but without labels. 因为这不会影响我们之后区分negative和positive，但是会对我们的model学习我们的data更有帮助。
* `label_for_lm()`是为了让我们的data和pretrain中的data有同一个形式的label。

```python
data_lm = (TextList.from_folder(path)
           #Inputs: all the text files in path
            .filter_by_folder(include=['train', 'test', 'unsup'])
           #We may have other temp folders that contain text files so we only keep what's in train and test, unsup is just all the text file without label
            .split_by_rand_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_for_lm()           
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))
data_lm.save('data_lm.pkl')
```

* This is a RNN neural network

  ```python
  learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
  # make drop_mult<1 to avoid underfitting.
  ```

* Interpret the accuracy here. It's actually saying you have around 30% to predict the next word right.

  ![image-20200207164714326](https://tva1.sinaimg.cn/large/0082zybpgy1gbopd4ma0qj30ui0u0460.jpg)

* fine tune好之后，模型包括两部分，一部分是理解语言，第二部分是预测下一个单词。我们的目的是为了之后做情感分析，所以我们的目的是去理解语言，所以只保存第一部分。也就是encoder的部分。

  ```python
  learn.save_encoder('fine_tuned_enc')
  ```

#### N-grams

N-grams使用的是词组出现的频率，再去判断新来的词是否为一个词组。对于同一个任务来说`I'd like to eat a hot ___`。正确答案是Dog，N-grams对于这一点是表现不好的，而Language Model可以通过知识的学习来预测出。So We don't have to worry about n-gram because we let the model to figure out what is the combination for the word.

#### Classifier Model

* 在读data时，要和language model里面产生的vocab一模一样，这样我们训练的language model才有用。
* 现在我们不用所有的data，因为我们要看label，不能作弊。

  \`\`\`python

  data\_clas = \(TextList.from\_folder\(path, vocab=data\_lm.vocab\)

  ```text
         #grab all the text files in path
         .split_by_folder(valid='test')
         #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
         .label_from_folder(classes=['neg', 'pos'])
         #label them all with their folders
         .databunch(bs=bs))
  ```

data\_clas.save\('data\_clas.pkl'\)

```text
* 使用的是语言分类模型。
```python
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
```

* 为了使用我们之前的语言模型去理解

  ```python
  learn.load_encoder('fine_tuned_enc')
  ```

* 只要有了用所有text去训练的language model，之后的classifier可以任意create，并且时间也不会很久。
* 对于语言分类，Jeremy发现一层一层unfreeze表现会更加好。

  ```python
  learn.freeze_to(-2) # unfreeze the top 2 layers.
  learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
  learn.freeze_to(-3)
  learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
  learn.unfreeze()
  learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
  ```

* `moms`是动量的意思。在RNN中，降低动量对训练有帮助。

