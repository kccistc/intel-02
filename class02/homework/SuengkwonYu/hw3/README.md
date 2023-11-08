# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset:	87
./splitted_dataset/train:	69
./splitted_dataset/train/success:	37
./splitted_dataset/train/fail:	32
./splitted_dataset/val:	18
./splitted_dataset/val/success:	10
./splitted_dataset/val/fail:	8

```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|54|52.48|8|0.0071|-|
|EfficientNet-B0|1.0|209|22.39|16|0.0049|-| 
|DeiT-Tiny|1.0|53|35.15|16|0.0001|-| 
|MobileNet-V3-large-1x|1.0|260|17.84|16|0.0058|-|


## FPS 측정 방법
inferencing을 하는 시간을 측정하여 그 시간의 역수가 초당 처리할 수 있는 inferencing의 갯수가 되고 이를 초당 프레임(FPS)이라고 볼 수 있다.
inferencing 하는 시간을 측정할 때는 모델을 불러오고 장치에 로드하는 시간을 뺀 순수히 모델을 거쳐 확률이 나오는 시간을 측정한다.
