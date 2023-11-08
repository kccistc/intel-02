# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
user3198352@iot07:~/workspace/otx-failpass/MNV3-task$ ds_count ./splitted_dataset 2
./splitted_dataset:	807
./splitted_dataset/train:	645
./splitted_dataset/train/fail:	305
./splitted_dataset/train/pass:	340
./splitted_dataset/val:	162
./splitted_dataset/val/fail:	76
./splitted_dataset/val/pass:	86
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.0 |45|46s|16|2.900e-03|-|
|EfficientNet-B0| 1.0 |136|1m 16s|16|2.450e-03|-|
|DeiT-Tiny| 1.0 |41|2m 41s|16|5.000e-05|-|
|MobileNet-V3-large-1x| 1.0 |232|5m 20s|8|0.00355|-|


## FPS 측정 방법
inferencing을 하는 시간을 측정하여 그 시간의 역수가 초당 처리할 수 있는 inferencing의 갯수가 되고 이를 초당 프레임(FPS)이라고 볼 수 있다.
inferencing 하는 시간을 측정할 때는 모델을 불러오고 장치에 로드하는 시간을 뺀 순수히 모델을 거쳐 확률이 나오는 시간을 측정한다.
