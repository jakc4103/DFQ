# DFQ
PyTorch implementation of [Data Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721).

## Results on classification task
- Tested with [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2)

<table>
<tr><th>ImageNet validation set (Acc.)   </th></tr>
<tr><td>

model/precision | FP32 | Int8*|
-----------|------|------|
Original   | 71.81 | 0.14
+ReLU | 71.78 | 0.15
+ReLU+LE | 71.78 | 70.39
+ReLU+BC  |  --  | 56.25
+ReLU+BC +clip_15  |  --  | 65.77
+ReLU+LE+BC  |  --  | 70.92

</td></tr> </table>

## Results on segmentation task
- Tested with [Deeplab-v3-plus_mobilenetv2](https://github.com/jfzhang95/pytorch-deeplab-xception)  
<table>
<tr><th>Pascal VOC 2012 val set (mIOU) </th><th>Pascal VOC 2007 test set (mIOU)</th></tr>
<tr><td>

model/precision | FP32  | Int8*|
----------------|-------|-------|
Original  | 70.81 |  59.71
+ReLU     | 70.72 |  60.17
+ReLU+LE  | 70.72 | 65.85
+ReLU+BC  |  --  |  68.8
+ReLU+BC +clip_15  |  --  | 65.5
+ReLU+LE+BC  |  --  | 69.14

</td><td>

model/precision | FP32  | Int8*  
----------------|-------|-------  
Original | 74.54 |  62.24
+ReLU    | 74.35 |  61.39
+ReLU+LE  | 74.35 | 69.55
+ReLU+BC  |  --  |  72.4
+ReLU+BC +clip_15  |  --  | 68.85
+ReLU+LE+BC  |  --  | 73.45

</td></tr> </table>

## Results on detection task  
- Tested with [MobileNetV2 SSD-Lite model](https://github.com/qfgaohao/pytorch-ssd)

<table>
<tr><th>Pascal VOC 2012 val set (mAP)   </th><th>Pascal VOC 2007 test set (mAP)  </th></tr>
<tr><td>

model/precision | FP32 | Int8*|
-----------|------|------|
Original   | 70.95 | 
+ReLU     | 67.44 | 
+ReLU+LE  | 67.44 | 
+ReLU+BC  |  --  |
+ReLU+BC +clip_15  |  --  |
+ReLU+LE+BC  |  --  |

</td><td>

model/precision | FP32  | Int8*  
----------------|-------|-------  
Original | 60.5 |  
+ReLU     | 57.61 |  
+ReLU+LE  | 57.61 | 
+ReLU+BC  |  --  |
+ReLU+BC +clip_15  |  --  |
+ReLU+LE+BC  |  --  |

</td></tr> </table>

## Usage
There are 5 arguments, all default to False
  1. quantize: whether to quantize parameters and activations.  
  2. relu: whether to replace relu6 to relu.  
  3. equalize: whether to perform cross layer equalization.  
  4. correction: whether to apply bias correction
  5. clip_weight: whether to clip weights in range [-15, 15] (for convolution and linear layer)

run the equalized model by:
```
python main_cls.py --quantize --relu --equalize
```

run the equalized and bias-corrected model by:
```
python main_cls.py --quantize --relu --equalize --correction
```

## Note
### Fake Quantization
  The 'Int8' model in this repo is actually simulation of 8 bits, the actual calculation is done in floating points.  
  This is done by quantizing-dequantizing parameters in each layer and activation between 2 consecutive layers;  
  Which means each tensor will have dtype 'float32', but there would be at most 256 (2^8) unique values in it.  
  ```
    Weight_quant(Int8) = Quant(Weight)
    Weight_quant(FP32) = Weight_quant(Int8*) = Dequant(Quant(Weight))
  ```

### 16-bits Quantization for Bias
  Somehow I cannot make **Bias-Correction** work on 8-bits bias quantization (even with data dependent correction).  
  I am not sure how the original paper managed to do it with 8 bits quantization, but I guess they either use some non-uniform quantization techniques or use more bits for bias parameters as I do.

## TODO
- [x] cross layer equalization
- [ ] high bias absorption
- [x] data-free bias correction
- [ ] test with detection model
- [x] test with classification model

## Acknowledgment
- https://github.com/jfzhang95/pytorch-deeplab-xception
- https://github.com/ricky40403/PyTransformer
- https://github.com/qfgaohao/pytorch-ssd
- https://github.com/tonylins/pytorch-mobilenet-v2
- https://github.com/xxradon/PytorchToCaffe
