# DFQ
PyTorch implementation of [Data Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721).

## Results on classification task
- Tested with [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2)
#### ImageNet validation set (Acc.)  
model/precision | FP32 | Int8*|
-----------|------|------|
Original   | 71.81 | 0.128
replace relu6 | 71.78 | 0.132
+layer equalization | 71.78 | 68.76

## Results on segmentation task
- Tested with [Deeplab-v3-plus_mobilenetv2](https://github.com/jfzhang95/pytorch-deeplab-xception)  
#### Pascal VOC 2012 val set (mIOU)  

model/precision | FP32  | Int8*|
----------------|-------|-------|
Original  | 70.81 |  56.99|
replace relu6  | 70.72 |  59.72|
+Layer equalization  | 70.72 | 65.97|  

#### Pascal VOC 2007 test set (mIOU)  
model/precision | FP32  | Int8*  
----------------|-------|-------  
Original | 74.54 |  59.48
replace relu6 | 74.35 |  60.15
+Layer equalization  | 74.35 | 69.43

## Results on detection task  
- Tested with [MobileNetV2 SSD-Lite model](https://github.com/qfgaohao/pytorch-ssd)
#### Pascal VOC 2012 val set (mAP)  
model/precision | FP32 | Int8*|
-----------|------|------|
Original   |  | 
replace relu6 |  | 
+layer equalization |  | 

#### Pascal VOC 2007 test set (mAP)  
model/precision | FP32  | Int8*  
----------------|-------|-------  
Original |  |  
replace relu6 |  |  
+Layer equalization  |  | 

## Usage
There are 3 arguments, all default to False
  1. quantize: whether to quantize parameters and activations.  
  2. relu: whether to replace relu6 to relu.  
  3. equalize: whether to perform cross layer equalization.  

You can run the equalized model by:
```
python main_cls.py --quantize --relu --equalize
```

## Note
The 'Int8' model in this repo is actually simulation of 8 bits, the actual calculation is done in floating points.  
This is done by quantizing-dequantizing parameters in each layer and activation between 2 consecutive layers;  
Which means each tensor will have dtype 'float32', but there would be at most 256 (2^8) unique values in it.  
```
  Weight_quant(Int8) = Quant(Weight)
  Weight_quant(FP32) = Weight_quant(Int8*) = Dequant(Quant(Weight))
```

## TODO
- [x] cross layer equalization
- [ ] high bias absorption
- [ ] data-free bias correction
- [ ] test with detection model
- [x] test with classification model

## Acknowledgment
- https://github.com/jfzhang95/pytorch-deeplab-xception
- https://github.com/ricky40403/PyTransformer
- https://github.com/qfgaohao/pytorch-ssd
- https://github.com/tonylins/pytorch-mobilenet-v2
- https://github.com/xxradon/PytorchToCaffe
