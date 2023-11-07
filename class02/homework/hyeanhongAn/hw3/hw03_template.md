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
|EfficientNet-V2-S|1.000|3.62979784|0:05:36.060029|8|lr: 3.550e-03|None|
|EfficientNet-B0|1.000 |7.828304905|0:01:14.406433|16|lr: 2.450e-03|None|
|DeiT-Tiny| 1.000|7.783286261|0:02:41.160913|4|lr: 2.450e-03|None|
|MobileNet-V3-large-1x|1.000 |6.930477183|0:00:47.957554|8|lr: 2.900e-03|None|


## FPS 측정 방법

```sh
def main():
    start = time.time()
    #inference time  start
    ##############################################
    
    #during code
    
    ##############################################
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    print("infer time: " +str(time.time()-start))
    #inference time cal
    return 0

```
- FPS time을 구하기 위해 inference time 나온값에 역수를 취해 준다. 
- FPS = (inference time)^-1
