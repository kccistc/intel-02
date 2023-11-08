# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 600
./splitted_dataset/train: 480
./splitted_dataset/train/FAIL: 240
./splitted_dataset/train/PASS: 240
./splitted_dataset/val: 120
./splitted_dataset/val/FAIL: 60​
./splitted_dataset/val/PASS: 60​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.0 | 2.5592111806358515 | 0:01:54.696882 | 16 | 0.0001 | epoch=12 |
|EfficientNet-B0| 1.0 | 7.010263927168069 | 0:02:22.663261 | 4 | 0.0005 | epoch=12 |
|DeiT-Tiny| 1.0 | 4.045830126034776 | 0:03:47.875395 | 8 | 0.0005 | epoch=11 |
|MobileNet-V3-large-1x| 1.0 | 7.753675056937502 | 0:01:13.686224 | 16 | 0.001 | epoch=14 |


## FPS 측정 방법
모델 로드 전
- fps = 1 / (end_time - start_time)
	- start_time : 모델 load 직전
	- end_time : prediction 직후

