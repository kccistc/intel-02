#!/usr/bin/env python3
"""
 Copyright (c) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from doctest import TestResults
import notebook_utils as utils
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy
import collections
import logging as log
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.runtime import Core, get_version

check = False
#################################################################################################3
utils_file_path = Path('../utils/notebook_utils.py')
notebook_directory_path = Path('.')
core = Core()
detection_model_path = "model/intel/person-detection-0202/FP16/person-detection-0202.xml"
reidentification_model_path = "model/intel/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml"
class Model:
    """
    This class represents a OpenVINO model object.

    """
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        """
        Initialize the model object

        Parameters
        ----------
        model_path: path of inference model
        batchsize: batch size of input data
        device: device used to run inference
        """
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        """
        Run inference

        Parameters
        ----------
        input: array of input data
        """
        result = self.compiled_model(input)[self.output_layer]
        return result
import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device
detector = Model(detection_model_path, device=device.value)
# since the number of detection object is uncertain, the input batch size of reid model should be dynamic
extractor = Model(reidentification_model_path, -1, device.value)
def preprocess(frame, height, width):
    """
    Preprocess a single image

    Parameters
    ----------
    frame: input frame
    height: height of model input data
    width: width of model input data
    """
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image


def batch_preprocess(img_crops, height, width):
    """
    Preprocess batched images

    Parameters
    ----------
    img_crops: batched input images
    height: height of model input data
    width: width of model input data
    """
    img_batch = np.concatenate([
        preprocess(img, height, width)
        for img in img_crops
    ], axis=0)
    return img_batch


def process_results(h, w, results, thresh=0.5):
    """
    postprocess detection results

    Parameters
    ----------
    h, w: original height and width of input image
    results: raw detection network output
    thresh: threshold for low confidence filtering
    """
    # The 'results' variable is a [1, 1, N, 7] tensor.
    detections = results.reshape(-1, 7)
    boxes = []
    labels = []
    scores = []
    for i, detection in enumerate(detections):
        _, label, score, xmin, ymin, xmax, ymax = detection
        # Filter detected objects.
        if score > thresh:
            # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
            boxes.append(
                [(xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h, (xmax - xmin) * w, (ymax - ymin) * h]
            )
            labels.append(int(label))
            scores.append(float(score))

    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])
    return np.array(boxes), np.array(scores), np.array(labels)


def draw_boxes(img, bbox, identities=None):
    """
    Draw bounding box in original image

    Parameters
    ----------
    img: original image
    bbox: coordinate of bounding box
    identities: identities IDs
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1.6,
            [255, 255, 255],
            2
        )
    return img


def cosin_metric(x1, x2):
    """
    Calculate the consin distance of two vector

    Parameters
    ----------
    x1, x2: input vectors
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
base_file_link = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/person_'
#image_indices = ['1_1.png', '1_2.png', '2_1.png']
#image_paths = [utils.download_file(base_file_link + image_index, directory='data') for image_index in image_indices]
#image1, image2, image3 = [cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) for image_path in image_paths]

# Define titles with images.
#data = {"Person 1": image1, "Person 2": image2, "Person 3": image3}

# Create a subplot to visualize images.
#fig, axs = plt.subplots(1, len(data.items()), figsize=(5, 5))

# Fill the subplot.
"""
for ax, (name, image) in zip(axs, data.items()):
    ax.axis('off')
    ax.set_title(name)
    ax.imshow(image)
"""
# Display an image.
#plt.show()
# Metric parameters
MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
#input_data = [image2, image3]
#img_batch = batch_preprocess(input_data, extractor.height, extractor.width)
#features = extractor.predict(img_batch)
#sim = cosin_metric(features[0], features[1])
#if sim >= 1 - MAX_COSINE_DISTANCE:
#    print(f'Same person (confidence: {sim})')
#else:
#    print(f'Different person (confidence: {sim})')
# Main processing function to run person tracking.

def run_person_tracking(frame):
    """
    Main function to run the person tracking:
    1. Create a video player to play with target fps (utils.VideoPlayer).
    2. Prepare a set of frames for person tracking.
    3. Run AI inference for person tracking.
    4. Visualize the results.

    Parameters:
    ----------
        source: The webcam number to feed the video stream with primary webcam set to "0", or the video path.
        flip: To be used by VideoPlayer function for flipping capture image.
        use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        skip_first_frames: Number of frames to skip at the beginning of the video.
    """

    player = None
    h, w = frame.shape[:2]
    input_image = preprocess(frame, detector.height, detector.width)
    # Measure processing time.
    start_time = time.time()
    # Get the results.
    output = detector.predict(input_image)
    stop_time = time.time()
    # processing_times.append(stop_time - start_time)
    # if len(processing_times) > 200:
    #     processing_times.popleft()

    _, f_width = frame.shape[:2]
    # Mean processing time [ms].
    #processing_time = np.mean(processing_times) * 1100
    #fps = 1000 / processing_time

    # Get poses from detection results.
    bbox_xywh, score, label = process_results(h, w, results=output)

    img_crops = []
    for box in bbox_xywh:
        x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
        img = frame[y1:y2, x1:x2]
        img_crops.append(img)

    # Get reidentification feature of each person.
    if img_crops:
        # preprocess
        img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
        features = extractor.predict(img_batch)
    else:
        features = np.array([])

    # Wrap the detection and reidentification results together
    bbox_tlwh = xywh_to_tlwh(bbox_xywh)
    detections = [
        Detection(bbox_tlwh[i], features[i])
        for i in range(features.shape[0])
    ]

    # predict the position of tracking target
    tracker.predict()

    # update tracker
    tracker.update(detections)

    # update bbox identities
    outputs = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        box = track.to_tlwh()
        x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
        track_id = track.track_id
        outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
    if len(outputs) > 0:
        outputs = np.stack(outputs, axis=0)

    # draw box for visualization
    if len(outputs) > 0:
        bbox_tlwh = []
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        frame = draw_boxes(frame, bbox_xyxy, identities)

    cv2.putText(
        img=frame,
        #text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
        text=f"hello",
        org=(20, 40),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=f_width / 1000,
        color=(0, 0, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return frame


NN_BUDGET = 100
MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
metric = NearestNeighborDistanceMetric(
    "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(
    metric,
    max_iou_distance=0.7,
    max_age=70,
    n_init=3
)
#################################################################################################

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from utils import crop
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier

import monitors
from helpers import resolution
from images_capture import open_images_capture

from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group('General')
    general.add_argument('-i', '--input', required=True,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    general.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    general.add_argument('-o', '--output',
                         help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    general.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    general.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    general.add_argument('--no_show', action='store_true',
                         help="Optional. Don't show output.")
    general.add_argument('--crop_size', default=(0, 0), type=int, nargs=2,
                         help='Optional. Crop the input stream to this resolution.')
    general.add_argument('--match_algo', default='HUNGARIAN', choices=('HUNGARIAN', 'MIN_DIST'),
                         help='Optional. Algorithm for face matching. Default: HUNGARIAN.')
    general.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    gallery = parser.add_argument_group('Faces database')
    gallery.add_argument('-fg', default='', help='Optional. Path to the face images directory.')
    gallery.add_argument('--run_detector', action='store_true',
                         help='Optional. Use Face Detection model to find faces '
                              'on the face images, otherwise use full images.')
    gallery.add_argument('--allow_grow', action='store_true',
                         help='Optional. Allow to grow faces gallery and to dump on disk. '
                              'Available only if --no_show option is off.')

    models = parser.add_argument_group('Models')
    models.add_argument('-m_fd', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Detection model.')
    models.add_argument('-m_lm', type=Path, required=True,
                        help='Required. Path to an .xml file with Facial Landmarks Detection model.')
    models.add_argument('-m_reid', type=Path, required=True,
                        help='Required. Path to an .xml file with Face Reidentification model.')
    models.add_argument('--fd_input_size', default=(0, 0), type=int, nargs=2,
                        help='Optional. Specify the input size of detection model for '
                             'reshaping. Example: 500 700.')

    infer = parser.add_argument_group('Inference options')
    infer.add_argument('-d_fd', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Detection model. '
                            'Default value is CPU.')
    infer.add_argument('-d_lm', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Facial Landmarks Detection '
                            'model. Default value is CPU.')
    infer.add_argument('-d_reid', default='CPU', choices=DEVICE_KINDS,
                       help='Optional. Target device for Face Reidentification '
                            'model. Default value is CPU.')
    infer.add_argument('-v', '--verbose', action='store_true',
                       help='Optional. Be more verbose.')
    infer.add_argument('-t_fd', metavar='[0..1]', type=float, default=0.6,
                       help='Optional. Probability threshold for face detections.')
    infer.add_argument('-t_id', metavar='[0..1]', type=float, default=0.3,
                       help='Optional. Cosine distance threshold between two vectors '
                            'for face identification.')
    infer.add_argument('-exp_r_fd', metavar='NUMBER', type=float, default=1.15,
                       help='Optional. Scaling ratio for bboxes passed to face recognition.')
    return parser


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        self.face_detector = FaceDetector(core, args.m_fd,
                                          args.fd_input_size,
                                          confidence_threshold=args.t_fd,
                                          roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(core, args.m_reid,
                                              match_threshold=args.t_id,
                                              match_algo=args.match_algo)

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug('Building faces database using images from {}'.format(args.fg))
        self.faces_database = FacesDatabase(args.fg, self.face_identifier,
                                            self.landmarks_detector,
                                            self.face_detector if args.run_detector else None, args.no_show)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        if self.allow_grow and len(unknowns) > 0:
            for i in unknowns:
                # This check is preventing asking to save half-images in the boundary of images
                if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                    (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                    (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                    continue
                crop_image = crop(orig_image, rois[i])
                name = self.faces_database.ask_to_save(crop_image)
                if name:
                    id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                    face_identities[i].id = id

        return [rois, landmarks, face_identities]


def draw_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
            global check
            check = True

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    return frame

def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]

def main():
    args = build_argparser().parse_args()
    global  check
    cap = open_images_capture(args.input, args.loop)
    frame_processor = FrameProcessor(args)

    frame_num = 0
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    input_crop = None
    if args.crop_size[0] > 0 and args.crop_size[1] > 0:
        input_crop = np.array(args.crop_size)
    elif not (args.crop_size[0] == 0 and args.crop_size[1] == 0):
        raise ValueError('Both crop height and width should be positive')
    video_writer = cv2.VideoWriter()

    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            if frame_num == 0:
                raise ValueError("Can't read an image from the input")
            break
        if input_crop is not None:
            frame = center_crop(frame, input_crop)
        if frame_num == 0:
            output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     cap.fps(), output_resolution):
                raise RuntimeError("Can't open video writer")

        detections = frame_processor.process(frame)
        presenter.drawGraphs(frame)
        frame = draw_detections(frame, frame_processor, detections, output_transform)
        metrics.update(start_time, frame)
        if(check == True ):
            run_person_tracking(frame)

        check = False
        frame_num += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frame_num <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            #cv2.resize(frame, frame, 960, 1280)
            cv2.imshow('Face recognition demo', frame)
            key = cv2.waitKey(1)
            # Quit
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
