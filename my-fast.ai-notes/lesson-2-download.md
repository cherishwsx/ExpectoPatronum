---
description: >-
  Original Detailed Notes:
  https://github.com/hiromis/notes/blob/master/Lesson2.md
---

# Lesson 2 Download

## General Things to Notice

### Overfitting

Training loss smaller then validation loss should be always the case, that means you are training your model correctly. As long as your loss and error is improving, that should be find. But if our error gets worse, that is really the sign of overfitting.

### Learning Rate

* **Rule of Thumb** is 3e-3 which is a default learning rate before you unfreeze. Next stage after unfreeze it and first part of slice is from `lr_finder` and second part of it is just devided by 10, 3e-4.
* After plotting your learning rate recorder, if you have a range that has small error, go for that range. But if you have a decreasing error rate,  then pick the range that decrease drastically.
* If `learn.lr_find()` not showing after you plot the recorder, try `learn.lr_find(start_lr=1e-5, end_lr=1e-1)` or find it again.
* And a _rule of thumb_ for chosing a good learning rate to do stage 2 training is $1\times10^{-4}$ to $3\times10^{-4}$.
* Why learning rate learner is different each time you run?

  \*\*\*\*

**Learning rate too high**

The `valid_loss` will be really high. 不管你经历了多少个epoch, 如果发生这种情况，就没有办法挽回了,你必须回去重新建立你的神经网络，从零开始适应一个较低的学习率（learning rate）。

![image-20200207161754903](https://tva1.sinaimg.cn/large/0082zybpgy1gbooimw7taj30zs05qwev.jpg)

![enter image description here](https://github.com/hiromis/notes/raw/master/lesson3/jose4.gif) 

It will jump back and forth and probably diverge at the end. ![enter image description here](https://github.com/hiromis/notes/raw/master/lesson3/jose5.gif)

#### Learning rate too low

![image-20200207161828460](https://tva1.sinaimg.cn/large/0082zybpgy1gbooj47l9vj30vs0u0mzp.jpg) 

1. We can see that the `error_rate` goes down very very slow. 

2. And `train_loss` is greater than `valid_loss` That means you didn't training the model enough and the number of epoch is not enough. So in this case, you might need to train the model in more epochs or with higher learning rate.

### Epoch

#### Too few epoch

`train_loss` is greater than `valid_loss` ![image-20200207161912308](https://tva1.sinaimg.cn/large/0082zybpgy1gboojstzqij30zu04sgma.jpg)

#### Too many eposh

Overfitting problem!

 ![image-20200207161942142](https://tva1.sinaimg.cn/large/0082zybpgy1gbookbsga8j30l60toaj7.jpg)

### Picures loaded in Computer

For grey scale, that basically is 1 dimension matrix, but if we have color image, we have 3 dimension. When you add an extra dimension, we call it a **tensor**, rather than a matrix. It's gonna be 3D tensor: red, green and blue.

 ![image-20200207124029466](https://tva1.sinaimg.cn/large/0082zybpgy1gboomklk8gj313a0o0n52.jpg) ![image-20200207124038978](https://tva1.sinaimg.cn/large/0082zybpgy1gboomz54jxj31280nzdsg.jpg)

### `error_rate`

* it's calculated from accuracy which is the proportion of how you predicted wrong
* And in the output it's always the validation `error_rate`.

**Linear Algebra Review**

**Matrix Vector Product**

Dot product: $\mathbf{y}=\mathbf{X}\mathbf{a}$

![](https://tva1.sinaimg.cn/large/0082zybpgy1gboonmundxj30jy09ojrp.jpg)

## Jupyter Tips

### 爬虫 Google Image Search

In Google Chrome press `Ctrl` `Shift` `j` on Windows/Linux and `Cmd` `Opt` `j` on macOS, and a small window the javascript 'Console' will appear. In Firefox press `Ctrl` `Shift` `k` on Windows/Linux or `Cmd` `Opt` `k` on macOS. That is where you will paste the JavaScript commands.

You will need to get the urls of each of the images. Before running the following commands, you may want to disable ad blocking extensions \(uBlock, AdBlockPlus etc.\) in Chrome. Otherwise the window.open\(\) command doesn't work. Then you can run the following commands:

```javascript
urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```

Then save it as txt file.

### Create directory based on labels

Define your `path` and `folder`. Notice that the `path` should be defined as `Path(string)` instead of only `string` for later concatenation.

```python
path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```

![image-20200207124133456](https://tva1.sinaimg.cn/large/0082zybpgy1gbooo3trq1j310e08umye.jpg)

## Create your own dataset

### `downlaod_image`

Dowlaod the url in the txt file you specified to the destination with max number of images.

### `verify_image`

It will try if every image in this folder can be opened and has n\_channels. If n\_channels is 3 – it'll try to convert image to RGB. If delete=True, it'll be removed it this fails. If resume – it will skip already existent images in dest. If max\_size is specified, image is resized to the same ratio so that both sizes are less than max\_size, using interp. Result is stored in dest, ext forces an extension type, img\_format and kwargs are passed to PIL.Image.save. Use max\_workers CPUs.

## Clean up your dataset

1. Create a databunch contains all the data using `ImageList`.
   * The difference between `ImageList` and `ImageDataButch` : You can’t use an ImageList to train a model \(it doesn’t have enough information specified for training - what’s the train/val split, where are the labels, etc.\). You have to go through the additional steps. The resulting ImageDataBunch will have all the information required. If you got confused because there is both `ImageDataBunch.from_folder` and `ImageList.from_folder` know that `ImageDataBunch.from_folder` is just a short cut using some default settings for all of the steps above! Take a look at the code here 45. It calls `ImageList.from_folder` internally as the very first step. 相当于ImageList是先前步骤。

     ```python
     @classmethod
     def from_folder(cls, path:PathOrStr, train:PathOrStr='train', valid:PathOrStr='valid', test:Optional[PathOrStr]=None,
                   valid_pct=None, seed:int=None, classes:Collection=None, **kwargs:Any)->'ImageDataBunch':
       "Create from imagenet style dataset in `path` with `train`,`valid`,`test` subfolders (or provide `valid_pct`)."
       path=Path(path)
       il = ImageList.from_folder(path, exclude=test)
       if valid_pct is None: src = il.split_by_folder(train=train, valid=valid)
       else: src = il.split_by_rand_pct(valid_pct, seed)
       src = src.label_from_folder(classes=classes)
       return cls.create_from_ll(src, test=test, **kwargs)
     ```

     And you can see that if we add more step to `ImageList`, it will be equivalent to the `ImageDataButch`:

```text
data = (ImageList.from_folder(mnist)
        .split_by_folder()
        .label_from_folder()
        .transform(tfms, size=32)
        .databunch()
        .normalize(imagenet_stats))

data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_folder()              #How to split in train/valid? -> use the folders
        .label_from_folder()            #How to label? -> depending on the folder of the filenames
        .add_test_folder()              #Optionally add a test set (here default name is test)
        .transform(tfms, size=64)       #Data augmentation? -> use tfms with a size of 64
        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch
```

2.  Fit a `cnn_learner` or other learner that you are using. And probably load a stage 2 weight if you previously trained the unclean data.

3.  Then you can do either validate your data or check duplicate.

* `DatasetFormatter().from_toplosses(learn_cln)` to have all sorted item from high to low loss for those that you have low confidence or high confidence but predicted wrong.
* `DatasetFormatter().from_similars(learn_cln)`

4.  Finally, you can `ImageCleaner(ds, idxs, path)` to call the widget to interact the images. And it will create a `cleaned.csv` instead of manipulating the original files.

5.  Then create a new DataButch using cleaned csv, and trian your model again.

## Inference

Sometimes, it's not often we have Google scaled computation so when we do inference which is you have pre-trained model and a image you want to test, you can do it in CPU. `defaults.device = torch.device('cpu')`.

## Putting the model in Production

#### `learn.export()`

This will create a file named 'export.pkl' in the directory where we were working that contains everything we need to deploy our model \(the model, the weights but also some metadata like the classes or the transforms/normalization used\). So the good thing is you don't need to do

```python

```

![image-20200207124145268](https://tva1.sinaimg.cn/large/0082zybpgy1gboooc8txqj3100078aby.jpg)

