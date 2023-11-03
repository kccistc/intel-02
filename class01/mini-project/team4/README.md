# Look at me
---------------------------
This example makes unknown faces blur except identified faces.

## Member
- Jinho Kim
- Eunyoung Kim

## Enviornment
- openvino_2023.1.0
- python venv .omz_venv
- source /opt/intel/openvino_2023.1.0/setupvars.sh

## Model Download and Convert
```sh
omz_downloader --list models.lst
omz_converter --list models.lst
```

## Using face_recognition_demo
- Percentage check
```sh
if identity.id != FaceIdentifier.UNKNOWN_ID:
            if percent < 70:
                identity.id = FaceIdentifier.UNKNOWN_ID

            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
```
- Make unknown blur
```sh
if identity.id == FaceIdentifier.UNKNOWN_ID:
            face_roi = frame[ymin:ymax, xmin:xmax]
            face_roi = cv2.GaussianBlur(face_roi, (0,0), 10)
            frame[ymin:ymax, xmin:xmax] = face_roi
```

## Make database
- make code that capture image on camera with python
- using these images to identify person

## Running
create run.sh file to run this code
```sh
#!/bin/bash

python ./face_recognition_demo.py \
  -i 4 \
  -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
  -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
  -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml \
  -t_fd 0.8 -t_id 0.8 \
  --verbose       \
  -fg "/home/kimjinho/face_gallery/captured_images"


./run.sh
```

## Result Image
![Result sample]("pictures/sample_result.png")