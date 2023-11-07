# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
““./tiny-task/splitted_dataset:	821
““./tiny-task/splitted_dataset/train:	656
““./tiny-task/splitted_dataset/train/pass:	347
““./tiny-task/splitted_dataset/train/fail:	309
““./tiny-task/splitted_dataset/val:	165
““./tiny-task/splitted_dataset/val/pass:	86
““./tiny-task/splitted_dataset/val/fail:	79
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.000|0.276s|5:46.363|8|lr: 3.550e-03|
|EfficientNet-B0| 1.000| 0.128s| 1:35.578|16|lr: 2.450e-03|
|DeiT-Tiny| 1.000|0.195s|2:41.944|16| lr: 5.000e-05|
|MobileNet-V3-large-1x| 1.000|0.143s|44.797|16|lr: 2.900e-03|


## FPS 측정 방법
hello_classification.py 코드 안에
def main 코드 시작 부분에  
```python
start = time.time()
```
을 추가한다
마지막 return 0 이전에
```python
print("Infer time : " +str (time.time()-start))
```
를 추가하여 inferencing 시간을 계산하였다

