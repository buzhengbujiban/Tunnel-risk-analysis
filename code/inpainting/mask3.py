from skimage import draw
from skimage import io
from PIL import Image
import numpy as np
import urllib.request
import json
import logging
import os
import pylab
import sys
import pylab
import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
annFile='/data/TsinghuaMOCS/instances_train.json'
coco=COCO(annFile)
#annFile = '/data/TsinghuaMOCS/medium_excavator.json'
#coco_kps=COCO(annFile)
catIds = coco.getCatIds(catNms=['cat'])
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = imgIds[0])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread('/data/TsinghuaMOCS/instances_train/instances_train/0018600.jpg')

plt.imshow(I); plt.axis('on')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])

plt.imshow(mask)
pylab.show()
