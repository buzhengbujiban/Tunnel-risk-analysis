# coding=gbk
import os 
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./instances_train.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='./train_labels', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)      # ��������ͨ�������4��

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))



if __name__ == '__main__':
    json_file =   arg.json_path # COCO Object Instance ���͵ı�ע
    ana_txt_save_path = arg.save_path  # �����·��

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)
    
    id_map = {} # coco���ݼ���id������������ӳ��һ���������
    for i, category in enumerate(data['categories']): 
        id_map[category['id']] = i

    # ͨ�����Ƚ���������ʱ�临�Ӷ�
    max_id = 0
    for img in data['images']:
        max_id = max(max_id, img['id'])
    # ע�����ﲻ��д�� [[]]*(max_id+1)�������б��ڵĿ��б����ַ
    img_ann_dict = [[] for i in range(max_id+1)] 
    for i, ann in enumerate(data['annotations']):
        img_ann_dict[ann['image_id']].append(i)

    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # ��Ӧ��txt���֣���jpgһ��
        f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
        '''for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))'''
        # �������ֱ�Ӳ��������ظ�����
        for ann_id in img_ann_dict[img_id]:
            ann = data['annotations'][ann_id]
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        f_txt.close()
        
# �ɰ棬����hhh
# """
# COCO ��ʽ�����ݼ�ת��Ϊ YOLO ��ʽ�����ݼ�
# --json_path �����json�ļ�·��
# --save_path ������ļ������֣�Ĭ��Ϊ��ǰĿ¼�µ�labels��
# """

# import os 
# import json
# from tqdm import tqdm
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--json_path', default='./instances_val2017.json',type=str, help="input: coco format(json)")
# parser.add_argument('--save_path', default='./labels', type=str, help="specify where to save the output dir of labels")
# arg = parser.parse_args()

# def convert(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = box[0] + box[2] / 2.0
#     y = box[1] + box[3] / 2.0
#     w = box[2]
#     h = box[3]

#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)

# if __name__ == '__main__':
#     json_file =   arg.json_path # COCO Object Instance ���͵ı�ע
#     ana_txt_save_path = arg.save_path  # �����·��

#     data = json.load(open(json_file, 'r'))
#     if not os.path.exists(ana_txt_save_path):
#         os.makedirs(ana_txt_save_path)
    
#     id_map = {} # coco���ݼ���id������������ӳ��һ���������
#     with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
#         # д��classes.txt
#         for i, category in enumerate(data['categories']): 
#             f.write(f"{category['name']}\n") 
#             id_map[category['id']] = i
#     # print(id_map)

#     for img in tqdm(data['images']):
#         filename = img["file_name"]
#         img_width = img["width"]
#         img_height = img["height"]
#         img_id = img["id"]
#         head, tail = os.path.splitext(filename)
#         ana_txt_name = head + ".txt"  # ��Ӧ��txt���֣���jpgһ��
#         f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
#         for ann in data['annotations']:
#             if ann['image_id'] == img_id:
#                 box = convert((img_width, img_height), ann["bbox"])
#                 f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
#         f_txt.close()