import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import requests
#
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image

# 预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
image_path = "/home/user1/code/forVGG/train/0/1.jpg"
img = Image.open(image_path).convert('RGB')
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    #json_file = open('./class_indices.json', 'r')
    class_indict = ['person','cat','dog']
except Exception as e:
    print(e)
    exit(-1)

# create model
## 导入预训练好的VGG16网络
vgg = models.vgg16(pretrained=True)
print(vgg)
model = vgg(num_classes=3)
# load model weights
model_weight_path = "/home/user1/code/vgg16-397923af.pth"
model.load_state_dict(torch.load(model_weight_path))

# 关闭 Dropout
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))     # 将输出压缩，即压缩掉 batch 这个维度
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()

