# DFQ
PyTorch implementation of [Data Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721) with some ideas from [ZeroQ: A Novel Zero Shot Quantization Framework](https://arxiv.org/abs/2001.00281).

## Results
Int8**: Fake quantization; 8 bits weight, 8 bits activation, 16 bits bias  
Int8*: Fake quantization; 8 bits weight, 8 bits activation, 8 bits bias  
Int8': Fake quantization; 8 bits weight(symmetric), 8 bits activation(symmetric), 32 bits bias  
Int8: Int8 Inference using [ncnn](https://github.com/Tencent/ncnn); 8 bits weight(symmetric), 8 bits activation(symmetric), 32 bits bias  

### On classification task
- Tested with [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2) and [ResNet-18](https://pytorch.org/docs/stable/torchvision/models.html)
- ImageNet validation set (Acc.)
<table>
<tr><th>MobileNetV2   </th><th>ResNet-18</th></tr>
<tr><td>

model/precision | FP32 | Int8** | Int8* | Int8' | Int8<br>(FP32-69.19)
-----------|------|------| ------ | ------|------
Original   | 71.81 | 0.102 | 0.1 | 0.062 | 0.082
+ReLU | 71.78 | 0.102 | 0.096 | 0.094 | 0.082
+ReLU+LE | 71.78 | 70.32 | 68.78 | 67.5 | 65.21
+ReLU+LE +DR | -- | 70.47 | 68.87 | -- | --
+BC  |  --  | 57.07 | 0.12 | 26.25 | 5.57
+BC +clip_15  |  --  | 65.37 | 0.13 | 65.96 | 45.13
+ReLU+LE+BC  |  --  | 70.79 | 68.17 | 68.65 | 62.19
+ReLU+LE+BC +DR  |  --  | 70.9 | 68.41 | -- | --

</td><td>

model/precision | FP32 | Int8** | Int8* 
-----------|------|------|------
Original   | 69.76 | 69.13 | 69.09 
+ReLU | 69.76 | 69.13 | 69.09 
+ReLU+LE | 69.76 | 69.2 | 69.2 
+ReLU+LE +DR | -- | 67.74 | 67.75 
+BC  |  --  | 69.04 | 68.56 
+BC +clip_15  |  --  | 69.04 | 68.56 
+ReLU+LE+BC  |  --  | 69.04 | 68.56 
+ReLU+LE+BC +DR  |  --  | 67.65 | 67.62

</td></tr> </table>

### On segmentation task
- Tested with [Deeplab-v3-plus_mobilenetv2](https://github.com/jfzhang95/pytorch-deeplab-xception)  
<table>
<tr><th>Pascal VOC 2012 val set (mIOU) </th><th>Pascal VOC 2007 test set (mIOU)</th></tr>
<tr><td>

model/precision | FP32  | Int8**| Int8*
----------------|-------|-------|------
Original  | 70.81 |  60.03 | 59.31
+ReLU     | 70.72 |  60.0 | 58.98
+ReLU+LE  | 70.72 | 66.22 | 66.0
+ReLU+LE +DR | -- | 67.04 | 67.23 
+ReLU+BC  |  --  |  69.04 | 68.42
+ReLU+BC +clip_15  |  --  | 66.99 | 66.39
+ReLU+LE+BC  |  --  | 69.46 | 69.22
+ReLU+LE+BC +DR  |  --  | 70.12 | 69.7

</td><td>

model/precision | FP32  | Int8** | Int8*
----------------|-------|-------|-------
Original | 74.54 |  62.36 | 61.21
+ReLU    | 74.35 |  61.66 | 61.04
+ReLU+LE  | 74.35 | 69.47 | 69.6
+ReLU+LE +DR | -- | 70.28 | 69.93
+BC  |  --  |  72.1 | 70.97
+BC +clip_15  |  --  | 70.16 | 70.76
+ReLU+LE+BC  |  --  | 72.84 | 72.58
+ReLU+LE+BC +DR  |  --  | 73.5 | 73.04

</td></tr> </table>

### On detection task  
- Tested with [MobileNetV2 SSD-Lite model](https://github.com/qfgaohao/pytorch-ssd)

<table>
<tr><th>Pascal VOC 2012 val set (mAP with 12 metric)   </th><th>Pascal VOC 2007 test set (mAP with 07 metric)  </th></tr>
<tr><td>

model/precision | FP32 | Int8**|Int8*
-----------|------|------|------
Original   | 78.51 | 77.71 | 77.86
+ReLU     | 75.42 | 75.74 | 75.58
+ReLU+LE  | 75.42 | 75.32 | 75.37
+ReLU+LE +DR | -- | 74.65 | 74.32
+BC  |  --  |  77.73 | 77.78
+BC +clip_15  |  --  | 77.73 | 77.78
+ReLU+LE+BC  |  --  | 75.66 | 75.66
+ReLU+LE+BC +DR  |  --  | 74.92 | 74.65

</td><td>

model/precision | FP32  | Int8** | Int8*
----------------|-------|-------|-------
Original | 68.70 |  68.47 | 68.49
+ReLU     | 65.47 | 65.36 | 65.56
+ReLU+LE  | 65.47 | 65.36 | 65.27
+ReLU+LE +DR | -- | 64.53 | 64.46
+BC  |  --  | 68.32 | 65.33
+BC +clip_15  |  --  | 68.32 | 65.33
+ReLU+LE+BC  |  --  | 65.63 | 65.58
+ReLU+LE+BC +DR  |  --  | 64.92 | 64.42

</td></tr> </table>

## Usage
There are 6 arguments, all default to False
  1. quantize: whether to quantize parameters and activations.  
  2. relu: whether to replace relu6 to relu.  
  3. equalize: whether to perform cross layer equalization.  
  4. correction: whether to apply bias correction
  5. clip_weight: whether to clip weights in range [-15, 15] (for convolution and linear layer)
  6. distill_range: whether to use distill data for setting min/max range of activation quantization

run the equalized model by:
```
python main_cls.py --quantize --relu --equalize
```

run the equalized and bias-corrected model by:
```
python main_cls.py --quantize --relu --equalize --correction
```

run the equalized and bias-corrected model with distilled data by:
```
python main_cls.py --quantize --relu --equalize --correction --distill_range
```

export equalized and bias-corrected model to onnx and generage calibration table file:
```
python convert_ncnn.py --equalize --correction --quantize --relu --ncnn_build path_to_ncnn_build_folder
```

## Note
### Distilled Data (2020/02/03 updated)
  According to recent paper [ZeroQ](https://github.com/amirgholami/ZeroQ), we can distill some fake data to match the statistics from batch-normalization layers, then use it to set the min/max value range of activation quantization.  
  It does not need each conv followed by batch norm layer, and should produce better and **more stable** results using distilled data (the method from DFQ sometimes failed to find a good enough value range).  

  Here are some modifications that differs from original ZeroQ implementation:
  1. Initialization of distilled data
  2. Early stop criterion

  ~~Also, I think it can be applied to optimizing cross layer equalization and bias correction. The results will be updated as long as I make it to work.~~   
  Using distilled data to do LE or BC did not perform as good as using estimation from batch norm layers, probably because of overfitting.

### Fake Quantization
  The 'Int8' model in this repo is actually simulation of 8 bits, the actual calculation is done in floating points.  
  This is done by quantizing-dequantizing parameters in each layer and activation between 2 consecutive layers;  
  Which means each tensor will have dtype 'float32', but there would be at most 256 (2^8) unique values in it.  
  ```
    Weight_quant(Int8) = Quant(Weight)
    Weight_quant(FP32) = Weight_quant(Int8*) = Dequant(Quant(Weight))
  ```

### 16-bits Quantization for Bias
  Somehow I cannot make **Bias-Correction** work on 8-bits bias quantization for all scenarios (even with data dependent correction).  
  I am not sure how the original paper managed to do it with 8 bits quantization, but I guess they either use some non-uniform quantization techniques or use more bits for bias parameters as I do.

### Int8 inference
  Refer to [ncnn](https://github.com/Tencent/ncnn), [pytorch2ncnn](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx), [ncnn-quantize](https://github.com/Tencent/ncnn/tree/master/tools/quantize), [ncnn-int8-inference](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference) for more details.  
  You will need to install/build the followings:  
  [ncnn](https://github.com/Tencent/ncnn)  
  [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)  
  
  Inference_cls.cpp only implements mobilenetv2. Basic steps are:  

  1. Run convert_ncnn.py to convert pytorch model (with layer equalization or bias correction) to ncnn int8 model and generate calibration table file. The name of out_layer will be printed to console.  
  ```
    python convert_ncnn.py --quantize --relu --equalize --correction
  ```
  
  2. compile inference_cls.cpp
  ```
    mkdir build
    cd build
    cmake ..
    make
  ```
  3. Inference! [link](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference)
  ```
    ./inference_cls --images=path_to_imagenet_validation_set --param=../modeling/ncnn/model_int8.param --bin=../modeling/ncnn/model_int8.bin --out_layer=name_from_step1
  ```

## TODO
- [x] cross layer equalization
- [ ] high bias absorption
- [x] data-free bias correction
- [x] test with detection model
- [x] test with classification model
- [x] use distilled data to set min/max activation range
- [ ] ~~use distilled data to find optimal scale matrix~~
- [ ] ~~use distilled data to do bias correction~~
- [x] True Int8 inference

## Acknowledgment
- https://github.com/jfzhang95/pytorch-deeplab-xception
- https://github.com/ricky40403/PyTransformer
- https://github.com/qfgaohao/pytorch-ssd
- https://github.com/tonylins/pytorch-mobilenet-v2
- https://github.com/xxradon/PytorchToCaffe
- https://github.com/amirgholami/ZeroQ
