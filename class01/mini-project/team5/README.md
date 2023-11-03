# Automatic Identification System
##### # Oh jonghyeok, Park jooeun

###### This application helps test supervisor to supervise exam candidate easier.
---

## Environment
```sh
ubuntu 22.04.3 LTS
Python 3.10.12
openvino 2023.1.0
```

## Installation
1. Download Open Model Zoo
```sh
git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
```

2. Install required packages
```sh
sudo apt install libopencv-dev
```

3. Create & enter python virtual environment
```sh
cd open_model_zoo
python3 -m venv .omz_venv
sourco .omz_venv/bin/activate
```

4. Update pip & install Required packages
```sh
python3 -m pip install --upgrade pip
pip install openvino openvino-dev
pip install -r demos/requirements.txt
```

## Used demo models
#### - face_recognition_demo
1. Preparing to Run
    ##### Enter to demo
    ```sh
    cd demos/face_recognition_demo/python
    ```

    ##### Download model
    ```sh
    omz_downloader --list models.lst
    ```

    ##### Convert model
    ```sh
    #if you don't have tensorflow, you should install
    pip install tensorflow
    omz_converter --list models.lst
    ```

    ##### Required model
    - face-detection-retail-0004
    - landmarks-regression-retail-0009
    - face-reidentification-retail
        > NOTE: you can skip installing the other model

2. Running
    ##### Arguments options
    You can check every options through the following usage message
    ```sh
    ./face_recognition_demo.py -h
    ```
    
    ##### Set up the environment
    ```sh
    source /opt/intel/openvino_2023/setupvars.sh
    ```

    ##### Create your gallery directory
    You should create your face database. The gallery is folder with images of persons. If you use only one image, the naming format should be {id}.jpg(e.g. Suzy.jpg). Also, there are allowed multiple images of the same person, but the naming format in that case should be {id}-{number of instance}.jpg(e.g. Suzy-0.jpg). When you run this model, you should use --rund_detector flag, then you can use face detector with your gallery. 

    ##### Run python code with arguments
    ```sh
    python3 face_recognition_demo.py 
    -i 0 -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml 
    -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml 
    -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml 
    --verbose 
    --run_detector 
    -fg "{your gallery directory}"
    ```


#### - text_spotting_demo
1. Preparing to Run
    ##### Enter to demo
    ```sh
    cd demos/text_spotting_demo/python
    ```

    ##### Download model
    ```sh
    omz_downloader --list models.lst
    ```

    ##### Convert model
    ```sh
    omz_converter --list models.lst
    ```
    
    ##### Required model
    - text-spotting-0005-detector
    - text-spotting-0005-recognizer-decoder
    - text-spotting-0005-recognizer-encoder
        > NOTE: you can skip installing the other model

2. Running
    ##### Arguments options
    You can check every options through the following usage message
    ```sh
    ./text_spotting_demo.py -h
    ```

    ##### Run python code with arguments
    ```sh
    python3 text_spotting_demo.py 
    -m_m intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml 
    -m_te intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml 
    -m_td intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml -
    i 0
    ```
    
    > NOTE: If you face the RuntimeError message, you should modify this code line.
    ```sh
    vi /open_model_zoo/demos/common/python/visualizers/instance_segmentation.py +79
    ```
    ```python
    def overlay_labels(self, image, boxes, classes, scores, texts=None):
        if texts:
            labels = texts
        elif self.labels:
            labels = (self.labels[class_id] for class_id in classes)
        else:
            #raise RuntimeError('InstanceSegmentationVisualizer must contain either labels or texts to display')
            return image
        template = '{}: {:.2f}' if self.show_scores else '{}' 
    ```

## Run
1. Create new directory
```sh
cd open_vino_zoo/demos
mkidr -p {directory}/{python}
cd {directory}/{python}
```
2. Copy two demo models
```sh
cp ../../face_recognition_demo/python/* .
cp ../../text_spotting_demo/python/* .
```
3. Run
```sh
./run.sh
```
