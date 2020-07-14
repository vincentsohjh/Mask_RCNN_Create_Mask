# Maintained by: Vincent Soh
# Adapted from visualise.py

import cv2
import numpy as np
import os
import sys
import random
import itertools
import colorsys
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
import PIL.Image
import time
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

# Load the pre-trained model data
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Change the config infermation
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()

# COCO dataset object names
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
    'teddy bear', 'hair drier', 'toothbrush'
]

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask):
    image[:, :, 0] = np.where(
        mask == 1,
        125,
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 1,
        12,
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 1,
        15,
        image[:, :, 2]
    )
    return image

#old function from visualise.py
# def apply_mask(image, mask, color=None, alpha=0.5):
    # """Apply the given mask to the image.
    # """
    # for c in range(3):
        # image[:, :, c] = np.where(mask == 1,
                                  # 0,
                                  # image[:, :, c])
                                  
    # return image


# This function is used to show the object detection result in original image. Now it also saves the mask created.
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    # colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    mask_total = np.zeros((masked_image.shape[0], masked_image.shape[1]), dtype=np.uint8)
    
    for i in range(N):
        label = class_names[class_ids[i]]
        color = "w"
        
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
            
        y1, x1, y2, x2 = boxes[i]
        square = (y2 - y1) * (x2 - x1)
        
       
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        if label == 'person':            

            # Label - not used to create masks
            # if not captions:
                # class_id = class_ids[i]
                # score = scores[i] if scores is not None else None
                # label = class_names[class_id]
                # x = random.randint(x1, (x1 + x2) // 2)
                # caption = "{} {:.3f}".format(label, score) if score else label
            # else:
                # caption = captions[i]
            # ax.text(x1, y1 + 8, caption,
                    # color='w', size=11, backgroundcolor="none")


            # Mask - Extract and add on to create an overall mask
                        
            mask = masks[:, :, i]
            mask_int = mask.astype(int)
            mask_total=mask_total + mask
                
            
            #for showing mask on image
            if show_mask:
                masked_image = apply_mask(masked_image, mask)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
                
                
    ax.imshow(masked_image.astype(np.uint8))

    if auto_show:
        plt.savefig("./mask_overlay.png")
        plt.show()
     
    mask_total = mask_total * 255

 
    
    
    cv2.imshow("mask",mask_total.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("mask_output.jpg",mask_total)

    return masked_image




if __name__ == "__main__":
    image = cv2.imread(sys.argv[1], -1)
    
    print("\n",image.shape,"\n")
    
    height, width, channels = image.shape
    results = model.detect([image], verbose=0)
    
    r = results[0]
    frame = display_instances(
         image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )

    cv2.imwrite('temp.png', image)
    

    image = PIL.Image.open("./temp.png")
    image.save("mask_pic.png")
    
