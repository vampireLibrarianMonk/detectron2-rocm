from detectron2.utils.logger import setup_logger

setup_logger()

# import common libraries
import cv2

# import some detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import requests

# URL of the image
url = 'http://images.cocodataset.org/val2017/000000439715.jpg'

# Send a GET request to the URL
response = requests.get(url)

import os

# Check if the request was successful
if response.status_code == 200:
    # Create saved directory
    os.makedirs("sample_images", exist_ok=True)

    # Write the content of the response to a file
    with open('sample_images/dude_on_horse.jpg', 'wb') as file:
        file.write(response.content)

cfg = get_cfg()
# add project-specific config here
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

im = cv2.imread("sample_images/dude_on_horse.jpg")
outputs = predictor(im)

# use the `Visualizer` to draw the predictions on the image
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Result", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
