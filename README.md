# 基于双域特征融合的图像去雾网络
>**摘要:**
图像去雾的目标是从有雾图像中恢复潜在的无雾图像. 先前的研究验证了利用清晰/退化图像对在空间域和频率域的差异进行去雾的有效性. 针对双域特征融合中空间域特征提取与融合不够充分, 频率域特征融合效果不佳等问题, 提出一种新颖的双域特征融合网络(dual-domain feature fusion network, DFFNet). 首先, 设计更适合图像软重建的空间域特征融合模块(spatial-domain feature fusion module, SFFM), 并引入大核注意力与像素注意力, 通过不同感受野分别对全局特征和局部特征进行建模. 同时, 设计频率域特征融合模块(frequency-domain feature fusion module, FFFM)使用卷积层来放大并丰富高频特征, 并利用通道交互来强调与融合多种高频特征. 结合这两种关键设计提出的DFFNet在两个基准数据集上展现出与最先进方法相当甚至更好的性能. DFFNet-L是第一个在SOTS-Indoor数据集上峰值信噪比(peak signal-to-noise ratio, PSNR)超过43dB的去雾方法, PSNR为43.83dB.
# Image Dehazing Network Based on Dual-domain Feature Fusion

>**Abstract:**
The goal of image dehazing is to restore the latent haze-free image from a hazy image. Previous research has validated the effectiveness of utilizing clear/degraded image pairs to address the differences in spatial-domain and frequency-domain for dehazing. To address the issues of insufficient spatial-domain feature extraction and fusion, as well as unsatisfactory frequency-domain feature fusion in dual-domain feature fusion, a novel net-work called Dual-domain Feature Fusion Network (DFFNet) is proposed. Firstly, a spatial-domain feature fusion module (SFFM) is designed that is more suitable for image soft reconstruction. Large kernel attention and pixel attention are introduced to model global and local features with different receptive fields. Secondly, a frequen-cy-domain feature fusion module (FFFM) is designed which amplifies and enriches high-frequency features using convolutional layers, and utilizes channel interaction to emphasize and fuse multiple high-frequency features. By combining these two key designs, the proposed DFFNet achieves comparable or even better performance than the state-of-the-art approaches on two benchmark datasets. DFFNet-L is the first dehazing method to achieve a peak signal-to-noise ratio (PSNR) exceeding 43dB on the SOTS-Indoor dataset, with a PSNR of 43.83dB.

## DFFNet框架:
![img.png](img.png)

## 实验结果:
![img_1.png](img_1.png)

## 数据集:
数据集的准备参考[Dehazeformer](https://github.com/IDKiro/DehazeFormer#vision-transformers-for-single-image-dehazing), 数据集的格式与Dehazeformer中的相同. 
请将数据集按照以下目录结构组织:

`Your path` <br/>
`├──RESIDE-IN` <br/>
     `├──train`  <br/>
          `├──GT`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──GT`  <br/>
          `└──hazy`  
`└──Haze4K` <br/>
     `├──train`  <br/>
          `├──GT`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──GT`  <br/>
          `└──hazy` 


## 预训练权重(weights)、实验结果(results)和消融实验的模型和权重(ablation_experiments):
[google drive](https://drive.google.com/drive/folders/1kMDQ7F9MjaakNh4TbCTrDoz023XxmKE_?usp=drive_link)\
[百度网盘](https://pan.baidu.com/s/1p8TFFrlsvITD10LJ1Kqg8g?pwd=0720)\
[夸克网盘](https://pan.quark.cn/s/b0385972c564)

## 准备环境
~~~
conda create -n DFFNet python=3.9
conda activate DFFNet
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
~~~

## 训练
~~~
# 在RESIDE-IN数据集上进行训练
python train_indoor.py --data_dir your_path --dataset RESIDE-IN --model DFFNet_L --gpu 0,1,2,3
# 在Haze4K数据集上进行训练
python train_haze4k.py --data_dir your_path --dataset Haze4K --model DFFNet_L --gpu 0,1,2,3
~~~

## 测试
~~~
# 在RESIDE-IN数据集上进行测试
python test_indoor.py --data_dir your_path --dataset RESIDE-IN --model DFFNet_L --saved_weight_dir weight_path
# 在Haze4K数据集上进行测试
python test_haze4k.py --data_dir your_path --dataset Haze4K --model DFFNet_L --saved_weight_dir weight_path
~~~

## 帮助:
如果您有任何问题, 请发邮件联系221027097@fzu.edu.cn或wwj20000720@163.com.

## 致谢
在此特别感谢我的导师、@[AmeryXiong](https://github.com/AmeryXiong)和@[c-yn](https://github.com/c-yn), 在这项工作中他们给了我很多帮助并无私地为我解答问题, 真诚地感谢你们!


