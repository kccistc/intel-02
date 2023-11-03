#!/bin/bash

python3 face_recognition_demo.py -i 0 -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml --verbose --run_detector -fg "/home/judy/Pictures/face_gallery" -m_m intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml -m_te intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml -m_td intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml

