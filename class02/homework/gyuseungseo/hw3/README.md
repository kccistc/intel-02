# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 701
./splitted_dataset/train: 506​
./splitted_dataset/train/false: 267​
./splitted_dataset/train/true: 293​
./splitted_dataset/val: 141
./splitted_dataset/train/false: 66​
./splitted_dataset/train/true: 75​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|0.9007|4.0336|04:30.089119|8|0.012|X
|EfficientNet-B0|1.0000|8.6646|00:59.131980|16|0.01|X
|DeiT-Tiny|1.0000|5.6412|02:50.453620|64|0.01|X
|MobileNet-V3-large-1x|1.0000|7.7499|00:39.443281|16|0.01|X


## FPS 측정 방법
파이썬 timer를 사용하여 이미지 하나가 로드하고부터 결과값이 나올 때 까지의 시간을 측정

```py
import time
```

```py
start_time = time.time()
    # Read input image
    image = cv2.imread(image_path)
```

```py

    log.info('')
    
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    log.info(f'FPS: {fps:.4f}')
```