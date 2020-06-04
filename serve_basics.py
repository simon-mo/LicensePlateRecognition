from ray import serve
import ray
from ray.serve.metric import PrometheusExporter

import time
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import requests
import os

os.environ["OMP_NUM_THREADS"] = "16"

import torch
from torchvision import transforms
import torchvision

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class ObjectDetector:
    def __init__(self):
        torch.set_num_threads(16)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True, progress=True).eval()
        self.preprocessor = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
        ])

    @serve.accept_batch
    def __call__(self, flask_request, data=None):
        if serve.context.web:
            data = [r.data for r in flask_request]
        tensors = []
        for single_image in data:
            pil_image = Image.open(BytesIO(single_image))
            input_tensor = self.preprocessor(pil_image)
            tensors.append(input_tensor)

        with torch.no_grad():
            output_tensors = self.model(tensors)

        output = []
        seen_labels = set()
        for output_tensor in output_tensors:
            labels = output_tensor["labels"]
            scores = output_tensor["scores"].numpy().tolist()
            names = [
                COCO_INSTANCE_CATEGORY_NAMES[l]
                for l in labels.numpy().tolist()
            ]
            output.append({"label": names[0], "score": scores[0]})
        return output


class ALPRServable:
    def __init__(self):
        from alpr import Recognizer
        self.recognizer = Recognizer()

    def __call__(self, flask_request, data=None):
        if serve.context.web:
            data = flask_request.data

        nparr = np.fromstring(data, np.uint8)
        frame = cv2.imdecode(
            nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        result = self.recognizer.evaluate(frame)

        return result


servables = [
    ("alpr", ALPRServable),
    ("object", ObjectDetector),
]

ray.init(address="auto")
serve.init(metric_exporter=PrometheusExporter)

serve.create_endpoint("object", "/object", methods=["POST"])
serve.create_backend("object:v1", ObjectDetector, config={"max_batch_size": 2})
serve.set_traffic("object", {"object:v1": 1})

serve.create_endpoint("alpr", "/alpr", methods=["POST"])
serve.create_backend("alpr:v1", ALPRServable)
serve.set_traffic("alpr", {"alpr:v1": 1})




