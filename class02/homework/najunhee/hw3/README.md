# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 480
./splitted_dataset/train: 96​
./splitted_dataset/train/<class#>: 47​
./splitted_dataset/train/<class#>: 49​
./splitted_dataset/val: 384
./splitted_dataset/train/<class#>: 189​
./splitted_dataset/train/<class#>: 195
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1| 3.995| 0:03:58.247765|8|0.0035
|EfficientNet-B0| 0.9895| 8.713| 0:00:51.812153|16|0.00245
|DeiT-Tiny| 1|5.031| 0:01:58.240250| 16| 5e-05|
|MobileNet-V3-large-1x|0.989|7.8|0:00:34.079570| 16| 0.0029|

## FPS 측정 방법

* import time 
* Get start_time before compile a model
* Get end_time after predict result
* (formula) FPS= 1/(end_time-start-time)
