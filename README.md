# MODEL CONVERSION CODES FROM TORCH TO TF1 WITH ONNX

> **requirements**  

- tensorflow==1.15.3  
- onnx==1.7.0  
- onnx-tf==1.7.0 from https://github.com/onnx/onnx-tensorflow/tree/tf-1.x **(should install onnx-tf for tf1)**  
- torch==1.7.1  
- torchvision==0.8.2  
- numpy  

> **instruction**  

***convert_wrn_tf1.py***  
a model converting file from torch to tf1.  
use wideresnet trained with cifar10 as default. change a model architecture as you want.

> **references**  

[wideresnet from MPL-torch repo.](https://github.com/kekmodel/MPL-pytorch)