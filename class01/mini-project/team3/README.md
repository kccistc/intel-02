# 3조 미니 프로젝트
## 차량 번호판 인식
---
## 프로젝트 설명
### object_detection을 이용하여 트럭을 인식하고 text_detection을 이용하여 번호판을 인식하려 한다. 추후 여러 대의 트럭들이 있을 시, 모든 차량들이 앞뒤 차량의 번호판을 인식하여 스스로가 위치파악을 하도록 한다.
---
## 동작 순서
```sh
* object_detection
1. 물체 탐색기로 트럭을 탐지
2. 크롭된 이미지에서 번호판을 한번 더 크롭

* text_detection & recognation
1. 크롭된 번호판에서 차량 번호 추출
```
---
## Test Env
```sh
* Ubuntu 22.04 
* OpenVINO 2023.1.0
```
---
## Installation Guide
```sh
* Create python virtual env
    python3 -m venv .venv

* Activate & install required packages
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
    sudo dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb
    pip install --upgrade paddlepaddle
```
---
## How to Use
```sh
./excute.sh

1. 카메라가 켜지면 인식 시키고 싶은 트럭을 카메라에 비춘 뒤 C키를 눌러 이미지 캡쳐 ESC키 누름 (truck.jpg로 이미지 저장됨)

2. 잠시 후 캡쳐 된 이미지가 켜지면 ESC키를 누름 (output.txt로 텍스트 저장됨)
```
---
## Result
![object-detection](home/mini_prooject/result_img/object.png)
![text-recognation](home/mini_prooject/result_img/text.png)
![output](home/mini_prooject/result_img/output.png)
