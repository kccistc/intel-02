# Class01 HW04


1. 가상환경 만들기

```sh
python3 -m venv ./env

source env/bin/activate
```

2. 필요한 라이브러리 install
```sh
pip install -q "openvino>=2023.1.0"

pip install -q opencv-python requests tqdm

pip install ipython
```

3. 추가 라이브러리 및 모델 불러오기
```sh
wget https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py

wget -P ./model/ "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx"
```

4. 디렉토리에 맞게 코드 수정
* 옵션 설정
```py
options = 'CANDY'
```
* Path 설정
```py
model_path = Path(f"hw4/model/{options.lower()}-9.onnx")

ov.save_model(ov_model, f"hw4/model/{options.lower()}-9.xml")

ir_path = Path(f"hw4/model/{options.lower()}-9.xml")
onnx_path = Path(f"hw4/model/{model_path}")
```
