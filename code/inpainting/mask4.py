
from pycocotools.coco import COCO
import os
from PIL import Image
import pylab
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


coco = COCO('/data/TsinghuaMOCS/instances_train.json')
img_dir = '/data/TsinghuaMOCS/instances_train/instances_train'
image_id = 1

img = coco.imgs[image_id]
# loading annotations into memory...
# Done (t=12.70s)
# creating index...
# index created!
img
# {'license': 2,
#  'file_name': '000000000074.jpg',
#  'coco_url': # 'http://images.cocodataset.org/train2017/000000000074.jpg',
#  'height': 426,
#  'width': 640,
#  'date_captured': '2013-11-15 03:08:44',
#  'flickr_url': # 'http://farm5.staticflickr.com/4087/5078192399_aaefdb5074_z.jpg# ',
#  'id': 74}
image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
plt.imshow(image, interpolation='nearest')
plt.show()
plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
coco.showAnns(anns)
mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])

plt.imshow(mask)

pylab.show()
