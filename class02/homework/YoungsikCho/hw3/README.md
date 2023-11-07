# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
splitted_dataset:	470
splitted_dataset/train:	376
splitted_dataset/train/Okay:	179
splitted_dataset/train/Fail:	197
splitted_dataset/val:	94
splitted_dataset/val/Okay:	44
splitted_dataset/val/Fail:	50
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|DeiT-Tiny| 1.000|5.13|1분29초|16| lr: 0.0015|
|MobileNet-V3-large-1x| 1.000|7.04|4분49초|16| lr: 0.0029|
|EfficientNet-V2-S| 1.000|3.64|3분57초|16| lr: 0.00355|
|EfficientNet-B0| 1.000|7.94|1분23초|16| lr: 0.00245|


## FPS 측정 방법

hello_classification.py를 변경했습니다.
```
# 1. main이 시작되기 전에 time_count에 시작 시간을 받아옵니다.
time_count = time.time()
```

```
# 2. main이 종료되기 전에 시작 시간과 현재 시간의 차이를 받아옵니다. 이 차이가 FPS를 의미합니다.
print("[ INFO ] FPS"+ str(time.time()-time_count))
```
