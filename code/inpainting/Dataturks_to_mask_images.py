from skimage import draw
from skimage import io
import numpy as np
import urllib.request
import json
import logging
import os
import sys


###################  INSTALLATION NOTE #######################
##############################################################

## pip install scikit-image
## pip install numpy

###############################################################
###############################################################

#enable info logging.
logging.getLogger().setLevel(logging.INFO)

def poly2mask(blobs, c, path_to_masks_folder, h, w, label, idx):
    mask = np.zeros((h, w))
    for l in blobs:
        fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = 1
    io.imsave(path_to_masks_folder + "/" + str(c) + "_" + label + "_" + str(idx) + ".png", mask)


def convert_dataturks_to_masks(path_to_dataturks_annotation_json, path_to_original_images_folder, path_to_masks_folder):
    # make sure everything is setup.
    if (not os.path.isdir(path_to_original_images_folder)):
        logging.exception(
            "Please specify a valid directory path to download images, " + path_to_original_images_folder + " doesn't exist")
        return
    if (not os.path.isdir(path_to_masks_folder)):
        logging.exception(
            "Please specify a valid directory path to write mask files, " + path_to_masks_folder + " doesn't exist")
        return
    if (not os.path.exists(path_to_dataturks_annotation_json)):
        logging.exception(
            "Please specify a valid path to dataturks JSON output file, " + path_to_dataturks_annotation_json + " doesn't exist")
        return

    f = open(path_to_dataturks_annotation_json)
    train_data = f.readlines()
    train = []
    for line in train_data:
        data = json.loads(line)
        train.append(data)
    c = 0
    for objects in train:
        blobs = []
        classes = {}
        image = objects['content'][objects['content'].rfind('_') + 1:objects['content'].rfind('.')]
        # download the images from given url
        urllib.request.urlretrieve(objects['content'], path_to_original_images_folder + "/image" + str(c) + ".jpg")
        annotations = objects['annotation']

        for annot in annotations:
            blobs = []
            label = annot['label']
            if (label != ''):
                if label not in classes:
                    classes[label] = 0

                points = annot['points']
                h = annot['imageHeight']
                w = annot['imageWidth']
                x_coord = []
                y_coord = []
                l = []
                for p in points:
                    x_coord.append(p[0] * w)
                    y_coord.append(p[1] * h)
                shape = (h, w)
                l.append(x_coord)
                l.append(y_coord)
                l.append(shape)
                blobs.append(l)
                poly2mask(blobs, c, path_to_masks_folder, annot['imageHeight'], annot['imageWidth'], label,
                          classes[label])
                classes[label] += 1
        c += 1


if (len(sys.argv) < 4):
    print(
    "Please provide path to dataturks json file, path to store ground truth images and path to store mask images in this order.")
    exit(0)
convert_dataturks_to_masks(sys.argv[1], sys.argv[2], sys.argv[3])