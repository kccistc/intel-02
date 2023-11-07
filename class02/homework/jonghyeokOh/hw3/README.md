# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: ???
./splitted_dataset/train: ???​
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/val: ???
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/train/<class#>: ???​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|0.83|5.34|21.15|8|0.0071| |
|EfficientNet-B0| 1 |114.94 | 56.19 | 16 | 0.0025 |  |
|DeiT-Tiny|1|38.60 | 110.31 | 64 | 0.0001 | | 
|MobileNet-V3-large-1x| 1 |131.17 | 32.33 | 16 | 0.0029 | |


## FPS 측정 방법

