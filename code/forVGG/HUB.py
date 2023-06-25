import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer


class VGGClassifier(LightningModule):
    def __init__(self, num_classes=4):
        super().__init__()

        self.vgg = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16_bn', pretrained=True)
        self.vgg.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.vgg(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root='/path/to/train/directory', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

val_data = datasets.ImageFolder(root='/path/to/val/directory', transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

test_data = datasets.ImageFolder(root='/path/to/test/directory', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

model = VGGClassifier(num_classes=len(train_data.classes))
trainer = Trainer(gpus=1)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)


import numpy as np

# 加载测试集
test_dataset = datasets.ImageFolder(root='path/to/test/dataset', transform=transform)

# 创建数据加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 模型评估
model.eval()

# 预测并保存结果
with open('predictions.txt', 'w') as f:
    for batch_idx, (data, _) in enumerate(test_loader):
        # 前向传播
        output = model(data)
        # 计算预测结果
        pred = output.argmax(dim=1)
        # 将预测结果转换为numpy数组
        pred_np = pred.cpu().numpy()
        # 将预测结果写入文件
        for p in pred_np:
            f.write('{}\n'.format(p))