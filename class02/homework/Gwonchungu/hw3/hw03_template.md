# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset:	300
./splitted_dataset/val:	60
./splitted_dataset/val/0:	30
./splitted_dataset/val/1:	30
./splitted_dataset/train:	240
./splitted_dataset/train/0:	120
./splitted_dataset/train/1:	120
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1|3.15|2:58|16|0.0035|0
|EfficientNet-B0|1|6.82|43|16|0.0024|0
|DeiT-Tiny| 1|4.43|1:23|16|5e-05|0
|MobileNet-V3-large-1x|1|6.07|33|16|0.0029|0


## FPS 측정 방법
```
hello_classification.py
```
열기

def main 코드 시작 아래  
```python
start = time.time()
```
추가

return 0 이전
```python
print("Infer time : " +str (time.time()-start))
```
추가
```
1/Infer time 
```
inferencing 역수로 FPS를 측정한다