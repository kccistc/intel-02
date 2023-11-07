# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 189
./splitted_dataset/train: 151​
./splitted_dataset/train/fail: 75​
./splitted_dataset/train/pass: 76​
./splitted_dataset/val: 38
./splitted_dataset/train/fail: 18​
./splitted_dataset/train/pass: 20​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 0.9474|4.0379738426806195 |  0:00:42.318826| 8| 7.100e-03 |epoch 6
|EfficientNet-B0| 0.9474 |8.644235327092789 |0:00:17.275360| 8|4.900e-03 |epoch 8
|DeiT-Tiny| 0.9737| 4.171664377423721| 0:00:37.262989|4|1.000e-04| epoch 8
|MobileNet-V3-large-1x| 0.9211 | 7.698499140084541| 0:00:13.293058 | 8| 5.800e-03 | epoch 8


## FPS 측정 방법

이미지를 읽고 로드된 모델을 통해 처리하는 'classification.py'를 사용합니다. 시간을 확인하는 몇 가지 코드를 추가하고 이 시간은 총 처리 시간이므로 FPS를 찾기 위해 확인한 시간으로 1을 나누어서 구합니다.