import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

def get_coco_roi(ins_coco, key_coco, image_path, label, save_path):
    try:
        # 采用getCatIds函数获取"person"类别对应的ID
        ins_ids = ins_coco.getCatIds()[5]
        print("%s 对应的类别ID: %d" % (label, ins_ids))
    except:
        print("[ERROR] 请输入正确的类别名称")
        return

    # 获取某一类的所有图片集合, 比如获取包含dog的所有图片
    imgIds = ins_coco.getImgIds(catIds=ins_ids)
    print("包含 {} 的图片共有 {} 张".format(label, len(imgIds)))

    for img in imgIds:
        try:
            img_info = ins_coco.loadImgs(img)[0]  #
        except:
            continue

        annIds = ins_coco.getAnnIds(imgIds=img_info['id'], catIds=ins_ids, iscrowd=None)
        imgpath = os.path.join(image_path, img_info['file_name'])
        jpg_img = cv2.imread(imgpath, 1)
        print(img_info['file_name'],"\n")
        if jpg_img is None:
            continue

        for ann in annIds:
            outline = ins_coco.loadAnns(ann)[0]
            ann1=ins_coco.loadAnns(ann)
            ins_coco.showAnns(ann1)
            #plt.show()
            plt.savefig('n1.jpg')
            #cv2.imwrite('n2.jpg', ann1)
            # 只提取类别对应的标注信息
            if outline['category_id'] != ins_ids:
                continue

            # 对人同时使用关键点判断, 如果关键点中含有0的数量比较多, 代表这个人是不完整或姿态不好的
            #if outline['category_id'] == 1:
                #key_outline = key_coco.loadAnns(ann)[0]
                #if key_outline['keypoints'].count(0) >= 10:
                #    continue

            # 将轮廓信息转为Mask信息并转为numpy格式
            mask = ins_coco.annToMask(outline)
            mask = np.array(mask)

            # 复制并扩充维度与原图片相等, 用于后续计算
            mask_three = np.expand_dims(mask, 2).repeat(3, axis=2)

            jpg_img = np.array(jpg_img)

            # 如果mask矩阵中元素大于0, 则置为原图的像素信息, 否则置为黑色
            result = np.where(mask_three > 0, jpg_img, 0)

            # 如果mask矩阵中元素大于0, 则置为白色, 否则为黑色, 用于生成第4通道图像信息
            alpha = np.where(mask > 0, 255, 0)
            alpha = alpha.astype(np.uint8)  # 转换格式, 防止拼接时由于数据格式不匹配报错

            b, g, r = cv2.split(result)  # 分离三通道, 准备衔接上第4通道
            rgba = [b, g, r, alpha]  # 将三通道图片转化为四通道(背景透明)的图片
            dst = cv2.merge(rgba, 4)  # 拼接4个通道
            dst = dst[int(outline['bbox'][1]):int(outline['bbox'][1]+outline['bbox'][3]),
                              int(outline['bbox'][0]):int(outline['bbox'][0]+outline['bbox'][2])]

            name, shuifx = os.path.splitext(img_info['file_name'])
            imPath = os.path.join(save_path, name + "_%05d" % (int(annIds.index(ann))) + ".png")
            print("[INFO] 当前进度: %d /%d" % (imgIds.index(img), len(imgIds)))
            # cv2.imwrite(imPath, dst)
            cv2.imencode('.png', dst)[1].tofile(imPath)  # 保存中文路径的方法

if __name__ == "__main__":
    # 定义Coco数据集根目录
    coco_root = r"E:/003 Datasets/002 CoCo2017/"

    coco_data = ['train2017', 'val2017']

    # 定义需要提取的类别
    labels = ["large_excavator"]
    for data in coco_data:
        coco_path = {
            'image_path': os.path.join(coco_root, data),
            'instances_json_path': coco_root + r"/annotations/instances_%s.json" % data,
            'keypoints_json_path': coco_root + r"/annotations/person_keypoints_%s.json" % data
        }
        ins_coco = COCO("/data/TsinghuaMOCS/instances_train.json")  #coco_path['instances_json_path']
        key_coco = COCO("/data/TsinghuaMOCS/large_excavator.json")

        for label in labels:
            save_path = "/home/lws/cocoqian_55"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            get_coco_roi(ins_coco, key_coco, "/data/TsinghuaMOCS/instances_train/instances_train", label, save_path)

    print("[INFO] 抠图结束")
