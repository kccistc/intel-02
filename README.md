# 상공회의소 서울기술교육센터 인텔교육 2기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-02.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

### Git LFS(Large File System)
크기가 큰 바이너리 파일들은 LFS로 관리됩니다. 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
# Note bin size is 132 bytes before LFS pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin

```shell
# If not installed
sudo apt install git-lfs
git lfs pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)



## Team project

### Team: ZERO (Zㅔ발 Eyes Re-Open)
졸음운전 사고 방지를 위한 운전자 모니터링 시스템(DMS) 및 자율 주행 시스템.  

* Members
  | Name | Role |
  |----|----|
  | 이유림 | 팀장. 서버 및 통신 (DB 설계 / 통신 API / 컨테이너 환경 구축) |
  | 나준희 | 자율 주행 시스템 (엣지 디바이스 제어 및 주행) |
  | 박영문 | 자율 주행 시스템 (Segmentation / 주행 방향 결정 알고리즘) |
  | 유나영 | 자율 주행 시스템 (Segmentation / 주행 방향 결정 알고리즘) |
  | 정인성 | DMS 시스템 (AI Modeling) |
  | 조영식 | 모니터링 시스템 시각화(Qt) |
* Project Github : https://github.com/66yurimi99/Z-ERO.git
* 발표자료 : https://github.com/66yurimi99/Z-ERO/doc/slide.ppt


### Team: Board Maestro
a system that converts hand-written mathematical expression on board to calculated result.

* Members
  | Name | Role |
  |----|----|
  | Yeongdae Kim | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | Seokhyun Ahn | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | Hyunhong Ahn | Data preprocessing, subsystem의 connection을 구성하고 preprocessing을 책임진다. |
  | Chungu Gwon | AI modeling, 원하는 결과가 나오도록 AI model을 선택, training을 수행한다. |
  | Jungjae Han | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, inference를 수행한다. |
  | Jaebyeong Park | UI/HW design, 사용자 인터페이스/Hardware를 정의하고 구현한다. |
* Project Github : https://github.com/Intel-Edge-AI-SW-Developers-2nd-Team-1/BoardMaestro.git
* 발표자료 : https://github.com/Intel-Edge-AI-SW-Developers-2nd-Team-1/BoardMaestro/doc/slide.ppt


### Team: Watchdogs
거리에 있는 CCTV를 이용하여 트래픽에 따른 신호등 제어 및 각종 사건사고 감지


* Members
  | Name | Role |
  |----|----|
  | 장석환 | AI_modeling, 사고분석 ai 학습 및 개발1 |
  | 김승현 | AI_modeling, 사고분석 ai 학습 및 개발2 |
  | 김형은 | 문서 제작 및 ppt제작,발표|
  | 서규승 | AI_modeling, 교통통제 및 project maneger|
  | 조성우 | edge_device_control, 신호등 및 raspberry cam제어 |
* Project Github : https://github.com/dnfm257/cctv_ctrl.git
* 발표자료 : https://github.com/dnfm257/cctv_ctrl/blob/main/doc/cctv_ctrl_ppt.pptx
