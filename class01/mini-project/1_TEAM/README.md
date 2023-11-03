# Title : Target-Person-Tracking-System
Target Person Tracking System

On this project, our goal is to integrate face recognition and person tracking, both of which are provided by Openvino. If the programme detects and identifies a specific individual, the software then checks whether the match rate is 70% or higher in relation to a previously trained target. Once this threshold is met (i.e., when the programme confirms the person to be recognized), the software draws a square around the person and continues to track them. Additionally, we utilized data augmentation to train the programme with images of the South Korean actor Wonbin. The integration is presented in merge.py.


### Virtual Enviroment
```sh
python3 -m venv target_tracking
```
### How to run
```
python ./face_recognition_demo.py 
-i 0 
-m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml 
-m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml 
-m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml 
--verbose 
-fg "/home/ubuntu/face_gallery"

```
### requirements.txt 

### AI model 1.facial recognition
- The code snippet has code lines that obtain target individual's images from a directory. The software trains itself on the images for precision. 

### AI model 2.Object Tracking
- Person Tracking with OpenVINO
- Based on training results, this code snippet enables articulating the desirable individual when he/she is caught on camera by drawing a square around them. 

### Image augmentation
- For further precision, we added image augmentation to train the programme.

### Result img


![스크린샷 2023-11-03 14-58-07](https://github.com/simpleis6est/Target-Person-Tracking-System/assets/143490860/6eca9b82-f30b-4328-a7f5-96ed27bf3ad9)

not Wonbin

![스크린샷 2023-11-03 14-58-30](https://github.com/simpleis6est/Target-Person-Tracking-System/assets/143490860/8080bb3b-a9e5-48a0-9840-67bd2fe88f07)

![스크린샷 2023-11-03 15-01-30](https://github.com/simpleis6est/Target-Person-Tracking-System/assets/143490860/20e9c233-ebea-47ec-9708-d9af4be02cc0)

Wonbin
