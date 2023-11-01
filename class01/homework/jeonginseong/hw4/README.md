# 가상환경 설정

## venv
```sh
python3 -m venv hw4_env
```
* venv 라이브러리로 hw4 디렉토리에 hw4_env 가상환경 디렉토리 생성

```sh
source hw4_env/bin/activate
```
* Activate 파일로 가상환경 활성화

```sh
deactivate
```
* Deactivate로 실행 비활성화

## Notes

```sh
pip install -r requirements.txt
```
* ```requirements.txt``` 패키지 설치

```sh
python3 hw4_answer.py
```
* Mosaic으로 style transfer된 webcam 영상출력