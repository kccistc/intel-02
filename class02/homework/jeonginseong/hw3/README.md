# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 461
./splitted_dataset/train: 368​
./splitted_dataset/train/OK: 188​
./splitted_dataset/train/NG: 180
./splitted_dataset/val: 93
./splitted_dataset/val/OK: 48​
./splitted_dataset/val/NG: 45​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 0.9333 | 2.833322975408485 | 00:00:25 | 6 | 0.0071 | epoch = 1 |
|EfficientNet-B0| 0.9688 | 6.063244569279186 | 00:00:46 | 4 | 0.0049 | epoch = 6 |
|DeiT-Tiny| 0.8556 | 3.877776606701392 | 00:01:04 | 16 | 0.0001 | epoch = 8 |
|MobileNet-V3-large-1x| 0.9375 | 5.058156137845869 | 00:00:23 | 5 | 0.0058 | epoch = 4 |


## FPS 측정 방법
start_time = time.time() // before loading model
end_time = time.time() // after loading model
fps = 1 / (end_time - start_time)