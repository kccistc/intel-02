# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
./splitted_dataset:	488
./splitted_dataset/val:	98
./splitted_dataset/val/false:	47
./splitted_dataset/val/ok:	51
./splitted_dataset/train:	390
./splitted_dataset/train/false:	188
./splitted_dataset/train/ok:	202
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0|3.702288385066577|0:03:11.783967|16|0.0071|-|
|EfficientNet-B0|1.0|7.805810571137716|0:00:52.412271|16|0.00245|-| 
|DeiT-Tiny|1.0|5.135415021408256|0:01:53.853308|16|0.000005|-|
|MobileNet-V3-large-1x|1.0|6.967374981727332|0:00:40.744901 |16|2.900e-03|-|


## FPS 측정 방법
1. 시작부분에서 타임 함수로 시간을 받아서 저장  
    >start_time = time.time()

2. 끝부분에서 타임 함수로 시간을 받아서 start_time을 빼면 실행 시간을 게산할 수 있고, 1초를 시간으로 나누어 FPS값을 계산    
    >print("FPS는 " +str(1/(time.time()-start_time)))