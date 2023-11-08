# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 99
./splitted_dataset/train: 79
./splitted_dataset/train/pass: 43​
./splitted_dataset/train/fail: 936
./splitted_dataset/val: 20
./splitted_dataset/train/pass: 43​
./splitted_dataset/train/fail: 26​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|57|1.43|16|0.0071|-|
|EfficientNet-B0|1.0|208|0.22|16|0.0049|-|
|DeiT-Tiny|1.0|26|0.36|16|0.001|-|
|MobileNet-V3-large-1x|1.0|272|13.21|16|0.0058|-| 


## FPS 측정 방법
