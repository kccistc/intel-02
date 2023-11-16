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
<프로젝트 요약>  
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
