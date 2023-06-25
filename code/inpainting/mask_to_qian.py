import cv2
from PIL import Image
import numpy as np
for i in range(6934,12404):
    xx = i
    ss1 = "{0:0>7.0f}".format(xx + 1)
    yuantu = "/data/TsinghuaMOCS/instances_train/instances_train/"+ss1+".jpg"
    masktu1 = "/home/lzd/mask_black/"+ss1+".jpg"
    #masktu2 = "F:/mx_matting/test_pic/img_resize/erzhitu_bai/000082_croped83_2.png"

    #使用opencv叠加图片
    img1 = cv2.imread(yuantu)
    img2 = cv2.imread(masktu1)
    img2=255-img2
    #img3 = cv2.imread(masktu2)

    '''
    alpha = 1
    meta = 1 - alpha
    gamma = 0'''
    #cv2.imshow('img1', img1)
    #cv2.imshow('img2', img2)
    #image = cv2.addWeighted(img1,alpha,img2,meta,gamma)
    image1 = cv2.add(img1, img2)
    #image2 = cv2.add(img1, img3)

    #cv2.imshow('image', image1)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imwrite("/home/lws/mask_qian/qi"+ss1+".jpg",image1) #抠出前景
    #cv2.imwrite("F:/mx_matting/test_pic/img_resize/fugai_png/000082_croped83_b.png",image2)
