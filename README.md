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

### Team: !(맛있으면 0 kcal)
<프로젝트 요약>
음식을 인식하고 해당 음식의 영양정보와 칼로리를 보여주고 기록한다.
하루 내에 기록된 정보를 바탕으로 하루 권장 칼로리 대비 섭취 칼로리 계산을 통해 식단관리를 해준다.

* Members
  | Name | Role |
  |----|----|
  | 김진호 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 박주은 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 유승권 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
  | 오종혁 | Database 커넥션 및 pyQt 디자인을 관리한다. |
  | 김은영 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
* Project Github : https://github.com/kocharmPrj/0kcal.git
* 발표자료 : https://github.com/kocharmPrj/Intel-AI-Project/blob/main/0kcal.ppt
