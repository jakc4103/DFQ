# DFQ
PyTorch implementation of [Data Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721).

## Results on segmentation task
- Tested with [Deeplab-v3-plus_mobilenetv2](https://github.com/jfzhang95/pytorch-deeplab-xception)  
#### Pascal VOC 2012 val set

model/precision | FP32  | Int8|
----------------|-------|-------|
Original  | 70.81 |  -|
replace relu6  | 70.72 |  59.72|
+Layer equalization  | 70.72 | 65.97|  

#### Pascal VOC 2007 test set  
model/precision | FP32  | Int8  
----------------|-------|-------  
Original | 74.54 |  -
replace relu6 | 74.35 |  60.15
+Layer equalization  | 74.35 | 69.43

## TODO
- [x] cross layer equalization
- [ ] high bias absorption
- [ ] data-free bias correction
- [ ] data-dependent bias correction & activation min/max
- [ ] test with detection model
- [ ] test with classification model

## Acknowledgment
- https://github.com/jfzhang95/pytorch-deeplab-xception
- https://github.com/ricky40403/PyTransformer
- https://github.com/xxradon/PytorchToCaffe
