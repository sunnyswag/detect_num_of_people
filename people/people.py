"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Edit by HHQ

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import cv2
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "people"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU. 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch 100
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    #DETECTION_MIN_CONFIDENCE = 0.9

    #BACKBONE = "resnet50"

    #IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 640
    #TRAIN_ROIS_PER_IMAGE =150
    #MAX_GT_INSTANCES = 200
    #POST_NMS_ROIS_TRAINING = 2500
    #RPN_ANCHOR_RATIOS = [0.35, 1, 2]



############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

"""
def color_splash(image, mask):
    #Apply color splash effect.
    #image: RGB image [height, width, 3]
    #mask: instance segmentation mask [height, width, instance count]

    #Returns result image.
    
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
"""
"""
==================VISUALIZE==================
"""
random.seed(0)
N=90
brightness = 1.0
hsv = [(i / N, 1, brightness) for i in range(N)]
random.shuffle(hsv)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def class_color(id,prob):
    _hsv = list(hsv[id])
    # _hsv[2]=random.uniform(0.8, 1)
    _hsv[2]=prob
    color = colorsys.hsv_to_rgb(*_hsv)
    return color

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


def draw_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [num_instances, height, width]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # if not ax:
    #     _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.copy()
    for i in range(N):
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        # color = colors[i]
        #color = class_color(class_id,score*score*score*score)
        color = color = (255/255, 48/255, 48/255)
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                       alpha=0.7, linestyle="dashed",
        #                       edgecolor=color, facecolor='none')

        cv2.rectangle(masked_image, (x1, y1),(x2, y2), [int(x*255) for x in (color)],2)#4

        # Label
        #label = class_names[class_id]
        label = "smoking"
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "%s %d%%"%(label, int(score*100)) if score else label
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")

        yyy=y1 -16
        if yyy <0:
            yyy=0

        cv2.putText(masked_image, caption,
                   (x1, yyy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, #1.5
                   [int(x*255) for x in (color)],2)#4
        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)#0.5
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            # ax.add_patch(p)
            pts = np.array(verts.tolist(), np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(masked_image,[pts],True,[int(x*255) for x in (color)],2)#4
    return masked_image.astype(np.uint8)



def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    show = np.zeros([2,2],dtype = int)
    class_names = ['BG', 'people']

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print("Number of people {}".format(r['rois'].shape[0]))
        # Color splash
        #splash = color_splash(image, r['masks'])
        splash=draw_instances(image,r['rois'],r['masks'],r['class_ids'],BalloonConfig.NAME,r['scores'],BalloonConfig.NAME)
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    #不用渲染图片
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 1
        success = True
        while success:
            #print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                if count % 7500 == 0:
                    # OpenCV returns images as BGR, convert to RGB
                    image = image[..., ::-1]
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]
                    # Color splash
                    num = r['rois'].shape[0]
                    contn = count//7500
                    if num <= 5:
                        show = np.concatenate((show,[[contn,num+2]]),axis=0)
                    elif num <= 10:
                        show = np.concatenate((show,[[contn,num+4]]),axis=0)
                    elif num <= 15:
                        show = np.concatenate((show,[[contn,num+6]]),axis=0)
                    elif num <= 20:
                        show =np.concatenate((show,[[contn,num+10]]),axis=0)
                    print(show)

                count += 1
        vwriter.release()
    # matplot(np.delete(show,[0,1],axis=0))
    # elif video_path:
    #     import cv2
    #     # Video capture
    #     vcapture = cv2.VideoCapture(video_path)
    #     width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = vcapture.get(cv2.CAP_PROP_FPS)

    #     # Define codec and create video writer
    #     file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    #     vwriter = cv2.VideoWriter(file_name,
    #                               cv2.VideoWriter_fourcc(*'MJPG'),
    #                               fps, (width, height))

    #     count = 1
    #     success = True
    #     while success:
    #         #print("frame: ", count)
    #         # Read next image
    #         success, image = vcapture.read()
    #         if success:
    #             # OpenCV returns images as BGR, convert to RGB
    #             image = image[..., ::-1]
    #             # Detect objects
    #             r = model.detect([image], verbose=0)[0]
    #             # Color splash

    #             if count % 1500 == 0:
	   #              num = r['rois'].shape[0]
	   #              contn = count//1500
	   #              if num <= 5:
	   #              	show = np.concatenate((show,[[contn,num+2]]),axis=0)
	   #              elif num <= 10:
	   #              	show = np.concatenate((show,[[contn,num+4]]),axis=0)
	   #              elif num <= 15:
	   #              	show = np.concatenate((show,[[contn,num+6]]),axis=0)
	   #              elif num <= 20:
	   #              	show =np.concatenate((show,[[contn,num+10]]),axis=0)
	   #              print(show)

    #             splash = draw_instances(image,r['rois'],r['masks'],r['class_ids'],BalloonConfig.NAME,r['scores'],BalloonConfig.NAME)
    #             # RGB -> BGR to save image to video
    #             splash = splash[..., ::-1]
    #             # Add image to video writer
    #             vwriter.write(splash)
    #             count += 1
    #     vwriter.release()
    # matplot(np.delete(show,[0,1],axis=0))
    # print("Saved to ", file_name)


# def matplot(show):
# 	barlist=plt.bar(show[:,0],show[:,1])

# 	plt.xlim(0,show.shape[0]+1)
# 	plt.ylim(0,40)

# 	plt.xlabel("Time")
# 	plt.ylabel('Num of people')

# 	plt.xticks(show[:,0])
# 	plt.yticks([0,5,10,15,20,25,30,35,40])

# 	for x,y in zip(show[:,0],show[:,1]):
# 		print((x,y))
# 		if y <= 10 :
# 			barlist[x-1].set_color('limegreen')
# 		elif y <=20 :
# 			barlist[x-1].set_color('dodgerblue')
# 		elif y <= 30 :
# 			barlist[x-1].set_color('orange')
# 		else:
# 			barlist[x-1].set_color('m')

# 		plt.text(x,y+0.5,'%.0f'%y,ha='center',va='bottom')

# 	plt.savefig("plot.png")
# 	plt.show()

"""
==============PLOT=================
"""

import matplotlib as mpl

zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')

def matplot(show):
    fig,ax = plt.subplots()
    limegreen = np.zeros([2,2],dtype = int)
    dodgerblue = np.zeros([2,2],dtype = int)
    orange = np.zeros([2,2],dtype = int)
    m = np.zeros([2,2],dtype = int)
    for x,y in zip(show[:,0],show[:,1]):
        print((x,y))
        if y <= 10 :
            limegreen = np.concatenate((limegreen,[[x,y]]),axis = 0)
        elif y <=20 :
            dodgerblue = np.concatenate((dodgerblue,[[x,y]]),axis = 0)
        elif y <= 30 :
            orange = np.concatenate((orange,[[x,y]]),axis = 0)
        else :
            m = np.concatenate((m,[[x,y]]),axis = 0)
    limegreen = deletFT(limegreen)
    dodgerblue = deletFT(dodgerblue)
    orange = deletFT(orange)
    m = deletFT(m)
    limegreen1 = ax.bar(limegreen[:,0],limegreen[:,1],color='limegreen')
    dodgerblue1 = ax.bar(dodgerblue[:,0],dodgerblue[:,1],color='dodgerblue')
    orange1 = ax.bar(orange[:,0],orange[:,1],color='orange')
    m1 = ax.bar(m[:,0],m[:,1],color='m')
    ax.set_title('当前时段车厢内人数分布图',fontproperties=zhfont,size="20")
    ax.set_xlim(0,show.shape[0]+1)
    ax.set_ylim(0,40)
    ax.set_ylabel('人数',fontproperties=zhfont,size="15")
    ax.set_xlabel('2018-12-30',size="15")
    ax.set_xticks([1,6,12,18,24,30,36])
    ax.set_xticklabels((r'$06:20$',r'$06:50$',r'$07:20$',r'$07:50$',r'$08:20$',r'$08:50$',r'$09:20$'))
    ax.set_yticks([0,5,10,15,20,25,30,35,40])
    leg = ax.legend((limegreen1[0],dodgerblue1[0],orange1[0],m1[0]),('少','正常','较多','爆满'))
    for text in leg.texts : text.set_font_properties(zhfont)

    def autolabel(rects):
        for rect in rects :
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2.0,1.01*height,'%d'%int(height),ha='center',va='bottom')

    autolabel(limegreen1)
    autolabel(dodgerblue1)
    autolabel(orange1)
    autolabel(m1)

    plt.savefig("plot.png")
    plt.show()

def deletFT(abc):
    abc = np.delete(abc,[0,1],axis = 0)
    return abc

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
