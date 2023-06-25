from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
#%matplotlib inline
coco = COCO('/data/TsinghuaMOCS/instances_train.json')
img_dir = '/data/TsinghuaMOCS/instances_train/instances_train'
image_id = 5

img = coco.imgs[image_id]
image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
plt.imshow(image, interpolation='nearest')

cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
coco.showAnns(anns)
plt.show()
mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])


plt.imshow(mask)

plt.savefig('n1.jpg')
cv2.imwrite('n2.jpg', mask)

