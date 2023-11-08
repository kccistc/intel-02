# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 291
./splitted_dataset/train: 232​
./splitted_dataset/train/okay: 115​
./splitted_dataset/train/nope: 117​
./splitted_dataset/val: 59
./splitted_dataset/train/okay: 28​
./splitted_dataset/train/nope: 31​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 0.9661016949152542|3.439775487676261|0:00:16.266474|8|7.100e-03|epochs = 1|
|EfficientNet-B0| 0.9152542372881356|7.771979172457242|0:01:08.311368|4|2.450e-03|epochs = 21
|DeiT-Tiny| 0.9830508474576272|5.06439184008597|0:00:12.258447|4|1.000e-04|epochs = 1
|MobileNet-V3-large-1x|0.9491525423728814|6.935714521943314|0:00:12.592565|8|5.800e-03|epochs = 5| 


## FPS 측정 방법
FPS(Frames Per Second) is the number of frames that a model can generate in one second. The equation is 
```
FPS = 1 / ({end time}-{start time})
```
A higher FPS value indicates that the model can process input data more quickly and provide results at a faster rate.

For getting FPS, I use OpenVINO python file that is named "hello_classification.py". I add time() functions in that code.

```python
# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = ov.Core()

  + start = time.time()
# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_path}')
```

```python
# --------------------------- Step 7. Process output ------------------------------------------------------------------
    predictions = next(iter(results.values()))

    # Change a shape of a numpy.ndarray with results to get another one with one dimension
    probs = predictions.reshape(-1)

  + end = time.time()
    # Get an array of 10 class IDs in descending order of probability
```

```python
# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')

  + print("FPS: ", 1/(end-start))

    return 0
```