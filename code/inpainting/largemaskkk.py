from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
coco = COCO('/home/lws/instances_train_folder/instances_train.json')
img_dir = '/home/lws/instances_train_folder/instances_train/instances_train'
for xx in range(1):
    image_id = xx+1
    img = coco.imgs[image_id]
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
        mask =mask | coco.annToMask(anns[i])*i
    ss='/home/lws/instances_train_folder/masks/'+str(xx+1)+'.jpg'

    fig = plt.figure(frameon=False)
    plt.imshow(mask)
    # ax.axis(off)
    # fig = plt.gcf()
    fig.set_size_inches(image.shape[1]/ 100.0, image.shape[0]/ 100.0)  # 输出width*height像素
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    print(image.shape)
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(ss)
    mask.fill(0)