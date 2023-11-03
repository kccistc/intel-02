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

import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.runtime import Core, get_version
from scipy.special import softmax

from text_spotting_demo.tracker import StaticIOUTracker

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

from visualizers import InstanceSegmentationVisualizer

from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DEVICE_KINDS = ['CPU', 'GPU', 'HETERO']

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28
DICTIONARY_OF_HUMAN = ['suzy', 'bae', 'Jonghyeok', 'Oh', 'Jooeun', 'Park']
DICTIONARY_CHQ = [0] * len(DICTIONARY_OF_HUMAN)
Name_clnt = None

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
    
    args = parser.add_argument_group('Options')
    args.add_argument('-m_m', '--mask_rcnn_model',
                      help='Required. Path to an .xml file with a trained Mask-RCNN model with '
                           'additional text features output.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_te', '--text_enc_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(encoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_td', '--text_dec_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(decoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-d', '--device',
                    help='Optional. Specify the target device to infer on, i.e : CPU, GPU. '
                        'The demo will look for a suitable plugin for device specified '
                        '(by default, it is CPU). Please refer to OpenVINO documentation '
                        'for the list of devices supported by the model.',
                    default='CPU', type=str, metavar='"<device>"')
    args.add_argument('--trd_input_prev_symbol',
                      help='Optional. Name of previous symbol input node to text recognition head decoder part.',
                      default='prev_symbol')
    args.add_argument('--trd_input_prev_hidden',
                      help='Optional. Name of previous hidden input node to text recognition head decoder part.',
                      default='prev_hidden')
    args.add_argument('--trd_input_encoder_outputs',
                      help='Optional. Name of encoder outputs input node to text recognition head decoder part.',
                      default='encoder_outputs')
    args.add_argument('--trd_output_symbols_distr',
                      help='Optional. Name of symbols distribution output node from text recognition head decoder part.',
                      default='output')
    args.add_argument('--trd_output_cur_hidden',
                      help='Optional. Name of current hidden output node from text recognition head decoder part.',
                      default='hidden')
    args.add_argument('-trt', '--tr_threshold',
                      help='Optional. Text recognition confidence threshold.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('--no_track',
                      help='Optional. Disable tracking.',
                      action='store_true')
    args.add_argument('--delay',
                      help='Optional. Interval in milliseconds of waiting for a key to be pressed.',
                      default=0, type=int, metavar='"<num>"')
    args.add_argument('--show_boxes',
                      help='Optional. Show bounding boxes.',
                      action='store_true')   
    args.add_argument('--show_scores',
                      help='Optional. Show detection scores.',
                      action='store_true') 
    args.add_argument('--keep_aspect_ratio',
                      help='Optional. Force image resize to keep aspect ratio.',
                      action='store_true')
    args.add_argument('-pt', '--prob_threshold',
                      help='Optional. Probability threshold for detections filtering.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('-a', '--alphabet',
                      help='Optional. Alphabet that is used for decoding.',
                      default='  abcdefghijklmnopqrstuvwxyz0123456789')
    args.add_argument('-r', '--raw_output_message',
                      help='Optional. Output inference results raw values.',
                      action='store_true')
    return parser

def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


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
    exam_name = None
    cnt = 0
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            exam_name = text
            cnt += 1
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))
            
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

    return frame, exam_name

def center_crop(frame, crop_size):
    fh, fw, _ = frame.shape
    crop_size[0], crop_size[1] = min(fw, crop_size[0]), min(fh, crop_size[1])
    return frame[(fh - crop_size[1]) // 2 : (fh + crop_size[1]) // 2,
                 (fw - crop_size[0]) // 2 : (fw + crop_size[0]) // 2,
                 :]

def main():
    global DICTIONARY_OF_HUMAN
    global DICTIONARY_CHQ
    global Name_clnt
    firstName_clnt = None
    lastName_clnt = None
    
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    frame_processor = FrameProcessor(args)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()
    
    # Read IR
    log.info('Reading Mask-RCNN model {}'.format(args.mask_rcnn_model))
    mask_rcnn_model = core.read_model(args.mask_rcnn_model)

    input_tensor_name = 'image'
    try:
        n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
        if n != 1:
            raise RuntimeError('Only batch 1 is supported by the demo application')
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))

    required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    for output_tensor_name in required_output_names:
        try:
            mask_rcnn_model.output(output_tensor_name)
        except RuntimeError:
            raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
                ', '.join(required_output_names)))

    log.info('Reading Text Recognition Encoder model {}'.format(args.text_enc_model))
    text_enc_model = core.read_model(args.text_enc_model)

    log.info('Reading Text Recognition Decoder model {}'.format(args.text_dec_model))
    text_dec_model = core.read_model(args.text_dec_model)

    mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name=args.device)
    mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()
    log.info('The Mask-RCNN model {} is loaded to {}'.format(args.mask_rcnn_model, args.device))

    text_enc_compiled_model = core.compile_model(text_enc_model, args.device)
    text_enc_output_tensor = text_enc_compiled_model.outputs[0]
    text_enc_infer_request = text_enc_compiled_model.create_infer_request()
    log.info('The Text Recognition Encoder model {} is loaded to {}'.format(args.text_enc_model, args.device))

    text_dec_compiled_model = core.compile_model(text_dec_model, args.device)
    text_dec_infer_request = text_dec_compiled_model.create_infer_request()
    log.info('The Text Recognition Decoder model {} is loaded to {}'.format(args.text_dec_model, args.device))

    hidden_shape = text_dec_model.input(args.trd_input_prev_hidden).shape
    text_dec_output_names = {args.trd_output_symbols_distr, args.trd_output_cur_hidden}

    if args.no_track:
        tracker = None
    else:
        tracker = StaticIOUTracker()

    if args.delay:
        delay = args.delay
    else:
        delay = int(cap.get_type() in ('VIDEO', 'CAMERA'))
    
    visualizer = InstanceSegmentationVisualizer(show_boxes=args.show_boxes, show_scores=args.show_scores)

    frames_processed = 0
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
            if frames_processed == 0:
                raise ValueError("Can't read an image from the input")
            break
        if input_crop is not None:
            frame = center_crop(frame, input_crop)
        if frames_processed == 0:
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
        
        
        
        if not args.keep_aspect_ratio:
            # Resize the image to a target size.
            scale_x = w / frame.shape[1]
            scale_y = h / frame.shape[0]
            input_image = cv2.resize(frame, (w, h))
        else:
            # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
            scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                           (0, w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w)).astype(np.float32)

        # Run the MaskRCNN model.
        mask_rcnn_infer_request.infer({input_tensor_name: input_image})
        outputs = {name: mask_rcnn_infer_request.get_tensor(name).data[:] for name in required_output_names}

        # Parse detection results of the current request
        boxes = outputs['boxes'][:, :4]
        scores = outputs['boxes'][:, 4]
        classes = outputs['labels'].astype(np.uint32)
        raw_masks = outputs['masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > args.prob_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_masks):
            mask = segm_postprocess(box, raw_mask, frame.shape[0], frame.shape[1])
            masks.append(mask)

        texts = []
        for feature in text_features:
            input_data = {'input': np.expand_dims(feature, axis=0)}
            feature = text_enc_infer_request.infer(input_data)[text_enc_output_tensor]
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            text_confidence = 1.0
            for i in range(MAX_SEQ_LEN):
                text_dec_infer_request.infer({
                    args.trd_input_prev_symbol: np.reshape(prev_symbol_index, (1,)),
                    args.trd_input_prev_hidden: hidden,
                    args.trd_input_encoder_outputs: feature})
                decoder_output = {name: text_dec_infer_request.get_tensor(name).data[:] for name in text_dec_output_names}
                symbols_distr = decoder_output[args.trd_output_symbols_distr]
                symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                text_confidence *= symbols_distr_softmaxed[prev_symbol_index]
                if prev_symbol_index == EOS_INDEX:
                    break
                text += args.alphabet[prev_symbol_index]
                hidden = decoder_output[args.trd_output_cur_hidden]

            texts.append(text if text_confidence >= args.tr_threshold else '')

        if len(boxes) and args.raw_output_message:
            log.debug('  -------------------------- Frame # {} --------------------------  '.format(frames_processed))
            log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
            for box, cls, score, mask in zip(boxes, classes, scores, masks):
                log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

        # Get instance track IDs.
        masks_tracks_ids = None
        if tracker is not None:
            masks_tracks_ids = tracker(masks, classes)



        presenter.drawGraphs(frame)
        frame, exam_n= draw_detections(frame, frame_processor, detections, output_transform)


        
        frame = visualizer(frame, boxes, classes, scores, masks, masks_tracks_ids)
        # I ADD THIS LINE
        #Name_clnt = None
        #if firstName_clnt is None and lastName_clnt is None:
        #for tmp in texts:
            #print("CHK!!! : ",tmp, " ", end ="")
        print(exam_n, Name_clnt)
        if Name_clnt is None:
            for name_clnt in texts :
                name_clnt = name_clnt.replace(" ","")
                for idx in range(0, len(DICTIONARY_OF_HUMAN), 1):
                    #print("chk same..", DICTIONARY_OF_HUMAN[idx])
                    if name_clnt == DICTIONARY_OF_HUMAN[idx]:
                        DICTIONARY_CHQ[idx] = 1

            #if ( firstName_clnt is not None and lastName_clnt is not None):
                for idx in range(0,len(DICTIONARY_OF_HUMAN),2):
                    if DICTIONARY_CHQ[idx] == 1 and DICTIONARY_CHQ[idx + 1] == 1:
                        firstName_clnt = DICTIONARY_OF_HUMAN[idx]
                        lastName_clnt = DICTIONARY_OF_HUMAN[idx+1]
                        Name_clnt = firstName_clnt + " " + lastName_clnt
                        firstName_clnt = None
                        lastName_clnt = None
        else:
            if exam_n is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2
                font_color = (0,0,255)
                font_thickness = 5
                org = (70,70)
                if exam_n == Name_clnt:
                    frame = cv2.putText(frame, Name_clnt, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                else:
                    Name_clnt = None
                    for tmp in DICTIONARY_CHQ:
                        tmp = 0
                    frame = cv2.putText(frame, "No", org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            else:
                Name_clnt = None
                for tmp in DICTIONARY_CHQ:
                    tmp = 0
        
        
        
        
        metrics.update(start_time, frame)

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Face recognition demo', frame)
            key = cv2.waitKey(1)
            # Quit
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

        start_time = perf_counter()
        
    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
