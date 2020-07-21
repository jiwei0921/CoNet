# CoNet
Code repository for our paper entilted "Accurate RGB-D Salient Object Detection via Collaborative Learning"  accepted at ECCV 2020 (poster).

# Overall
![avatar](https://github.com/jiwei0921/CoNet/blob/master/overall111.png) 


## CoNet Code

### > Requirment
+ pytorch 1.0.0+
+ torchvision
+ PIL
+ numpy

### > Usage
#### 1. Clone the repo
```
git clone https://github.com/jiwei0921/CoNet.git
cd CoNet/
```
 
#### 2. Train/Test
+ test     
Our test datasets [link](https://github.com/jiwei0921/RGBD-SOD-datasets) and checkpoint [link](https://pan.baidu.com/s/1ceRpBrSjIxM0ut3t8awDfg) code is **12yn**. You need to set dataset path and checkpoint name correctly.        

'--phase' as **test** in demo.py   
'--param' as **True** in demo.py  
```
python demo.py
```

+ train     
Our training dataset [link](https://pan.baidu.com/s/1EMKE7pwLg70sfYvQQAB1kA) code is **203g**. You need to set dataset path and checkpoint name correctly.     

'--phase' as **train** in demo.py      
'--param' as **True or False** in demo.py        
Note: True means loading checkpoint and False means no loading checkpoint.      
```
python demo.py
```

### > Results  
![avatar](https://github.com/jiwei0921/CoNet/blob/master/Comparison.png)     
  
We provide [saliency maps](https://pan.baidu.com/s/1hQH89lhzgR3fk2Y3eI_Jww) (code: qrs2) of our CoNet on 8 datasets (DUT-RGBD, STEREO, NJUD, LFSD, RGBD135, NLPR, SSD, SIP) as well as 2 extended datasets (NJU2k and STERE1000) refer to CPFP_CVPR19.
+ Note:  For evaluation, all results are implemented on this ready-to-use [toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

  
### > Related RGB-D Saliency Datasets
All common RGB-D Saliency Datasets we collected are shared in ready-to-use manner.       
+ The web link is [here](https://github.com/jiwei0921/RGBD-SOD-datasets).


### If you think this work is helpful, please cite
```
@InProceedings{Wei_2020_ECCV,       
   author = {Wei {Ji} and Jingjing {Li} and Miao {Zhang} and Yongri {Piao} and Huchuan {Lu}},   
   title = {Accurate RGB-D Salient Object Detection via Collaborative Learning},     
   booktitle = "ECCV",     
   year = {2020}     
}  
```


+ Thanks for related authors to provide the code or results, particularly, [Deng-ping Fan](http://dpfan.net), [Hao Chen](https://github.com/haochen593), [Chun-biao Zhu](https://github.com/ChunbiaoZhu), etc. 

### Contact Us
More details can be found in [Github Wei Ji.](https://github.com/jiwei0921/)    
If you have any questions, please contact us ( weiji.dlut@gmail.com ).

