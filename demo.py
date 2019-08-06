# ae_h - 2019-08-06

import os
import random
import skimage.io

ROOT_DIR = os.path.abspath('./')

IMG_DIR = os.path.join(ROOT_DIR, 'temp_img')

COCO_MODEL = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'logs')

from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize


class DemoConfig(Config):
    NAME = 'democonf'

    STEPS_PER_EPOCH = 100

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 80


if __name__ == '__main__':
    config = DemoConfig()
    config.display()

    model = modellib.MaskRCNN(mode='inference', model_dir=OUTPUT_DIR, config=config)
    model.load_weights(COCO_MODEL, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    img_file = next(os.walk(IMG_DIR))[2]
    image = skimage.io.imread(os.path.join(IMG_DIR, random.choice(img_file)))

    # detection
    result = model.detect([image], verbose=1)
    r = result[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
