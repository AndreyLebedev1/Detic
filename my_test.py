import json
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import sys
import numpy as np
import os, json, cv2, random


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


sys.path.insert(0, 'third_party/CenterNet2/')


from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test


cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)



BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)


import matplotlib.pyplot as plt

im = cv2.imread("unn-deep-learning/my_photo.jpg")
# cv2.imshow("image", im)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()
# Change the model's vocabulary to a customized one and get their word-embedding
#  using a pre-trained CLIP model.

from detic.modeling.text.text_encoder import build_text_encoder


def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


vocabulary = 'my_test'
metadata = MetadataCatalog.get("__used")
metadata.thing_classes = ['computer_monitor', 'earphone', 'painting',
                          'computer_keyboard']  # Change here to try your own vocabularies!
classifier = get_clip_embeddings(metadata.thing_classes)
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)
# Reset visualization threshold
output_score_threshold = 0.65
for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
    predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = output_score_threshold



outputs = predictor(im)
v = Visualizer(im[:, :, ::-1], metadata)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow("image", out.get_image()[:, :, ::-1])
# cv2.waitKey(3000)
# cv2.destroyAllWindows()
cv2.imwrite("unn-deep-learning/my_photo_detected.jpg", out.get_image()[:, :, ::-1])

# save results of test
import json

#create namespace
names = [metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]
#create boxes
arr = outputs["instances"].pred_boxes

np_arr = []
i = 0
result = {}
for elem in arr:
    np_arr = elem.detach().cpu().numpy().tolist()
    result[names[i]] = []
    for elem in np_arr:
        result[names[i]].append(round(elem))
    i += 1
# print(f"your result: \n{result}")


with open("unn-deep-learning/true_result.json", "r") as fh:
    true_result = json.load(fh)

# with open("true_result.json", "w") as fh:
#     json.dump(result, fh)

if true_result == result:
    print("Test passed! Very good, nice!")
else:
    print("Test not passed!")