import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

# 初始化根目录
train_path = '/home/user1/yolo/jiafei_projects/yolo_model/predict_res/1'
test_path = '/home/user1/yolo/jiafei_projects/yolo_model/predict_res/1'

# 定义读取文件的格式
# 数据集
class MyDataSet(Dataset):
    def __init__(self, data_path:str, transform=None):
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.transform = transforms
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx:int):
        img_path = self.path_list[idx]
        #print(img_path.split('.'))
        if img_path.split('.')[0] == '0':
            label = 0
        if img_path.split('.')[0]=='1':
            label = 1
        if img_path.split('.')[0]=='2':
            label = 2
        if img_path.split('.')[0]=='3':
            label = 3
        labels = torch.as_tensor(label, dtype=torch.int64)
        img_path = os.path.join(self.data_path, img_path)
        img = Image.open(img_path)
        img = self.transform(img)
        return img, labels

    def __len__(self)->int:
        return len(self.path_list)

train_ds = MyDataSet(train_path)

full_ds = train_ds
train_size = int(0.8*len(full_ds))
test_size = len(full_ds) - train_size
new_train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

# 数据加载
new_train_loader = DataLoader(new_train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
new_test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

