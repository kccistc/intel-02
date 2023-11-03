#!/bin/bash

python ./face_recognition_demo.py \
  -i 4 \
  -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
  -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
  -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml \
  -t_fd 0.8 -t_id 0.8 \
  --verbose       \
  -fg "/home/kimjinho/face_gallery/captured_images"

