# MINI PROJECT

## 0. Contributor
- Youngmoon Park
- Junhee Na
- Youngsik Cho

## 1. Model
- Interactive Face Detection C++ Demo
- Instance Segmentation Python* Demo

## 2. Supported Models
- Interactive Face Detection C++ Demo
    * age-gender-recognition-retail-0013
    * anti-spoof-mn3
    * emotions-recognition-retail-0003
    * face-detection-adas-0001
    * face-detection-retail-0004
    * face-detection-retail-0005
    * face-detection-retail-0044
    * facial-landmarks-35-adas-0002
    * head-pose-estimation-adas-0001

- Instance Segmentation Python* Demo

    * instance-segmentation-person-0007
    

## 3. Seqeunce
 1. 랜덤으로 주어질 키워드를 설정해줍니다.

 2. 5초 후 카메라에 나오는 얼굴들의 감정을 분석하고 랜덤으로 주어진 키워드의 감정표현과 가장 유사한 사람을 우승자로 선발합니다.

 3. 종료된 이미지를 segmentation 적용을 통해 사람을 감지합니다. 

 ## 4. How to run


- Interactive Face Detection C++ Demo
```
./interactive_face_detection_demo -i <path_to_video>/<input_video>.mp4 -m <path_to_model>/face-detection-adas-0001.xml --mag <path_to_model>/age-gender-recognition-retail-0013.xml --mhp <path_to_model>/head-pose-estimation-adas-0001.xml --mem <path_to_model>/emotions-recognition-retail-0003.xml --mlm <path_to_model>/facial-landmarks-35-adas-0002.xml --mam <path_to_model>/anti-spoof-mn3.xml -d GPU
```
<br>

- Instance Segmentation Python* Demo

```
instance_segmentation_demo.py [-h] -m MODEL [--adapter {openvino,ovms}] -i INPUT [-d DEVICE] --labels LABELS [-t PROB_THRESHOLD] [--no_track] [--show_scores]
                                     [--show_boxes] [--layout LAYOUT] [-nireq NUM_INFER_REQUESTS] [-nstreams NUM_STREAMS] [-nthreads NUM_THREADS] [--loop] [-o OUTPUT]
                                     [-limit OUTPUT_LIMIT] [--no_show] [--output_resolution OUTPUT_RESOLUTION] [-u UTILIZATION_MONITORS] [-r]

```
## 5. Result Image

https://potent-thing-08e.notion.site/Result-Image-8a7121ab875f4aad856329d0463a37bc?pvs=4

