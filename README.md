# YOLOv8-FWR
A lightweight model for underwater ROI region target detection. 

**Environment Dependencies Setting**  

This setup is for Windows 11 and NVIDIA-supported GPU.
1.	Install Python 3.8.19, 64 bit
2.	Pytorch Installation- Version 2.1.0+cu118
3.	Torchvision Installation- Version 0.16.0+cu118
4.	Opencv-python Installation- Version 4.7.0.72
5.	Updating GPU drives-Install NVIDIA drivers.
6.	CUDA installation- Cuda 11.8
7.	CuDnn Installation- Version 8.7.0 
8.	Numpy Installation- Version 1.23.0
9.	Pandas Installation- Version 2.0.3
10.	Scipy Installation- Version 1.10.1
11.	Scikit-learn: Installation- Version 1.3.2
12.	Matplotlib Installation- Version 3.7.5
13.	Cython Installation- Version 3.0.10
14.	Tensorboard Installation- Version 2.14.0
15.	Absl-py Installation- Version 2.1.0
16.	Thop Installation- Version 0.1.1.pos2209072238
17.	Torchsummary Installation- Version 1.5.1

**Data preparation**  
The labeled file of the image is in txt format.  
Create a yaml file to configure the dataset:  
 Example usage: yolo train data=coco128.yaml  
 parent  
 ├── ultralytics  
 └── datasets  
     └── data(Includes images, labels and yaml configuration files.)  
Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]  
path: ../datasets/data  # datset rooat dir  
train: images/train2017  # train images (relative to 'path')   
val: images/val2017  # val images (relative to 'path')   
test:  # test images (optional)  

Classes  
names:

**Training & validation**
1.	Training
```
yolo detect train data=.yaml model=yolov8-FWR.yaml epochs=200 batch=32 imgsz=640 val=True patience=50 optimizer='AdamW'
```
2.	Validation
```
yolo detect val data=.yaml model=best.pt imgsz=640 batch=32 conf=0.0001'
```
# Descriptive
Underwater laser imaging often suffers from significant background noise in the form of noisy lightbars, which adversely affects the imaging process and subsequent 3D reconstruction. To address this issue, we propose a lightweight deep learning algorithm named YOLOv8-FWR specifically designed to enhance the efficiency and quality of underwater laser imaging by accurately detecting lightbars in the ROI region. Our approach comprises three key components: (1) a novel Focal_SPPF pooling module to mitigate background noise interference; (2) a Weighted Feature Concat Module (WFCM) to improve the detection of small target lightbars, ensuring complete 3D reconstruction; and (3) an optimized C2f_Rep module to lightweight the backbone network, reducing parameters while maintaining accuracy. We developed a dataset for underwater scenarios and evaluated the improved model through ablation and comparison experiments. Results show that YOLOv8-FWR outperforms the original model by achieving an 8.6% improvement in mAP50-95 and reducing parameters by 18.3%. A favorable balance between detection accuracy and number of parameters is achieved. Additionally, experiments on public datasets (VOC2012 and UDD) validate its good generalizability.
# Datasets
The two publicly available datasets used in this study can be downloaded from [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#rights) and [UDD](https://github.com/chongweiliu/UDD_Official).
