import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

NUM_POINTS = 50
UNSURE_INTERNAL_REGION = 0.15
UNSURE_EXTERNAL_REGION = 0.25
IMG_SIZE = 1024

detect_model = YOLO('yolov9e.pt')
sam_model = FastSAM('FastSAM-x.pt')


def resize_image(image, larger_size):
    _shape = image.shape[:2]
    larger_dim = np.argmax(np.array(_shape))
    smaller_dim = 1 - larger_dim
    new_shape = [0, 0]
    new_shape[larger_dim] = larger_size
    new_shape[smaller_dim] = int(_shape[smaller_dim] * larger_size / _shape[larger_dim])
    new_shape.reverse()
    return cv2.resize(image, new_shape)


def check_point_in_object(point, bboxes):
    for bbox in bboxes:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if bbox[0] + w * UNSURE_INTERNAL_REGION <= point[0] <= bbox[2] - w * UNSURE_INTERNAL_REGION and bbox[1] + h * UNSURE_INTERNAL_REGION <= point[
            1] <= bbox[3] - h * UNSURE_INTERNAL_REGION:
            return 1
        elif bbox[0] - w * UNSURE_EXTERNAL_REGION <= point[0] <= bbox[2] + w * UNSURE_EXTERNAL_REGION and bbox[1] - h * UNSURE_EXTERNAL_REGION <= \
                point[1] <= bbox[3] + h * UNSURE_EXTERNAL_REGION:
            return 2
    return 0


def object_detect(image, show_process=False):
    if show_process:
        _image_bboxes = image.copy()
        _image_points = image.copy()
    output = detect_model(image)
    bboxes = []
    for bbox in output[0].boxes.xyxy:
        _box = [int(i) for i in bbox]
        bboxes.append(_box)
        if show_process:
            x1, y1, x2, y2 = _box
            cv2.rectangle(_image_bboxes, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

    points = []
    point_label = []
    for i in range(NUM_POINTS):
        for j in range(NUM_POINTS):
            point = [int(i * w / NUM_POINTS), int(j * h / NUM_POINTS)]
            point_stt = check_point_in_object(point, bboxes)
            if point_stt == 0:
                points.append(point)
                point_label.append(point_stt)
    for bbox in bboxes:
        points.append([int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
        point_label.append(1)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        for i in range(1, 5):
            points.append([int((x1 + x2) / 2), y1 + int((y2 - y1) / 5) * i])
            point_label.append(1)
            points.append([x1 + int((x2 - x1) / 5) * i, int((y1 + y2) / 2)])
            point_label.append(1)

    if show_process:
        for i in range(len(points)):
            cv2.circle(_image_points, points[i], radius=3, color=(0, 255, 0) if point_label[i] else (0, 0, 255), thickness=3)
        cv2.imshow("Object Bboxes", _image_bboxes)
        cv2.imshow("Object Points", _image_points)
    return bboxes, points, point_label


def run_sam_model(image, points, point_label, output_path, show_process=False):
    everything_results = sam_model(image, device='cpu', retina_masks=True, imgsz=1024, conf=0.3, iou=0.9)
    prompt_process = FastSAMPrompt(image, everything_results, device='cpu')

    ann = prompt_process.point_prompt(points=points, pointlabel=point_label)
    mask = np.array(ann[0].masks.data.numpy(), dtype=np.uint8)[0] * 255
    _mask = np.array(mask == 0, dtype=np.uint8) * 255
    masked_image = np.zeros_like(image, dtype=np.uint8)
    masked_image[mask == 255] = image[mask == 255]
    if show_process:
        cv2.imshow("Result", masked_image)
    cv2.imwrite(os.path.join(os.path.dirname(output_path), f"mask_{os.path.basename(output_path)}"), mask)
    cv2.imwrite(os.path.join(os.path.dirname(output_path), f"_mask_{os.path.basename(output_path)}"), _mask)
    cv2.imwrite(output_path, masked_image)
    return masked_image


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", default="./inputs/raw.png")
    parser.add_argument("-o", "--output-path", default="./outputs/raw.jpg")
    parser.add_argument("-sp", "--show-process", action="store_true", default=False)
    args = parser.parse_args()

    IMAGE_PATH = args.image_path
    OUTPUT_PATH = args.output_path
    SHOW_PROCESS = args.show_process

    image = cv2.imread(IMAGE_PATH)
    image = resize_image(image, IMG_SIZE)
    h, w, _ = image.shape
    bboxes, points, point_label = object_detect(image, SHOW_PROCESS)
    run_sam_model(image, points, point_label, OUTPUT_PATH, args.show_process)
    if args.show_process:
        cv2.waitKey()
