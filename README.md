# Hard-Aware-Deeply-Cascaded-Embedding_Only_Cars196

Run the code **only for Cars196 dataset**. Good reference for green hand in metric learning.
Pure bash & c++ code for training and extract features. Feel free to DIY your py script to build your test workflow.

---

# Hard-Aware-Deeply-Cascaded-Embedding [ICCV2017 (spotlight)]

**Congratulations to our work is accepted by ICCV2017 (spotlight).**

**There exist a typo previously about the initial learning rate of CARS196, 0.001 should be changed to 0.01. The HDC prefers larger learning rates.**

**Here is the link of the trained models(permitted for NON-COMMERCIAL usage only, if for COMMERCIAL, please contact yhyuan@pku.edu.cn)**

**Drop Box [Stanford-Online-Products/CUB-200-2011/CARS196](https://www.dropbox.com/sh/jpku87vedyohy27/AACDNvAXM8q7kYel0npJ2IFZa?dl=0)** 

**BaiduYunPan [Stanford-Online-Products/CUB-200-2011/CARS196](https://pan.baidu.com/s/1chDg54)** 

**04/11/2017  Add some tips on how to add new layers to caffe!**

**Note!!!** I found many people do not know how to add new layers to caffe framework. Here is a vivid explainations: First you need to add the "*.hpp *cpp *.cu" to the project. Then you need to edit the caffe.proto. First you need to check the max ID that you have used. Here we will take the [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)(2018/08/06) as an example. You could check that in the `Layer type-specific parameters` block, the ID of these params must be unique. In my case I need to add `optional PairFastLossParameter pair_fast_loss_param = 148;` below `optional WindowDataParameter window_data_param = 129;` . Besides, you also need to add the the following lines to spercify the parameters of the newly added layers.

```
   message PairFastLossParameter {
     optional float margin = 1 [default = 1.0];
     optional float hard_ratio = 2 [default = 1.0];
     optional float factor = 3 [default = 10];
     enum MODE {
       POS = 0;
       NEG = 1;
       BOTH = 2;
     }
     optional MODE mode = 4 [default = BOTH];
   }

``` 


**Update 03/30/2017 Information** :  **The attached models are not well trained for the final test, as there exist some small bugs. We will release all the final single best models as soon as possible ! we fix a bug that the gradient problem for pair_fast_loss_layer !**

**Information** :  **The weight_decay for CARS196 should be 0.0002, the original version is typed to be 0.0005 !**

This repository has the source code for the paper "Hard-Aware-Deeply-Cascaded-Embedding_release". This paper is available
 on [arXiv](https://arxiv.org/abs/1611.05720). For the loss layer implementation, look at the folder caffe_layers.
 

## Citing this work
If you find this work useful in your research, please consider citing :

    @article{yuan2016HDC,
      title={Hard-Aware Deeply Cascaded Embedding},
      author={Yuan, Yuhui and Yang, Kuiyuan and Zhang, Chao},
      journal={arXiv preprint arXiv:1611.05720},
      year={2016}
    }
    
## Installation
1. Install [Caffe](https://github.com/BVLC/caffe) (including the python interface if you want to use the test code) 
2. Add the "NormalizationLayer" and "PairFastLossLayer" to the caffe.

     
## Prerequisites
1. matplotlib,cPickle,numpy,lmdb
2. We assumed that you are using Windows (you can rewrite the *.bat to *.sh if you choose Linux or MacOs)

## Datasets & Models 
1.  Download pretrained GoogLeNet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
2.  Download the datasets needed (you could download them from the official sets): [StanfordOnlineProducts](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)  [CARS196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) [CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) (we will release our results on both both in-shop and consumer2shop.)
3.  You should change the path of images in the Training Prototxt.

## Usage

**Use new tool `extract_features2file`**: 
Put `extract_features2file.cpp` to caffe `src/caffe/tools` dir and build for a new tool.

**Process Data**: (you should in the folder /src_code/):
(just generate for `cars196`)
```
   python hdc_process.py
```
   
**Training Models**: (you should in the root folder)
```
   bash train.sh
```
   **You could change the image path in the training prototxt to train the models with bounding boxes or not**
   
**Extract Features**:
 ```
   bash extract_features_once.sh
```
**Test Models**: (you should in the folder /src_code/):
```
   python test_new.py
```

**You need to modify the relative path to adjust your environment.**

If you do the things right, you can get the output below (after 4832 times iteration):

```
$ python test_new.py
start compute distances ...
finished compute distances ...
stanford cars mean recall@ 1 : 0.589718
stanford cars mean recall@ 2 : 0.708892
stanford cars mean recall@ 4 : 0.802361
stanford cars mean recall@ 8 : 0.876153
stanford cars mean recall@ 16 : 0.929283
stanford cars mean recall@ 32 : 0.963842
```


**Improve Space**

To get better results than the paper, you could simply by sampling more batches. For convenience, we all sample for 5000 times. In our experiments, by sampling 10,000 times could further improve the performance.
