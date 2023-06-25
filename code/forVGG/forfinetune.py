# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:28:40 2020

@author: zqq
"""

import json
from keras.applications import VGG16  # 导入VGG16，首次导入会进行下载，下载速度太慢就去我提供的连接进行下载
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from keras.models import load_model

train_dir = '/home/user1/code/forVGG/data/train/'  # 训练数据集
valid_dir = '/home/user1/code/forVGG/data/val'  # 验证数据集

# 加载VGG16模型
conv_base = VGG16(weights='imagenet',  # 使用预训练数据
                  include_top=False,  # 不使用原来的分类器
                  input_shape=(150, 150, 3))  # 将输入尺寸调整为(150, 150, 3)

# 添加自己的网络模型
model = models.Sequential()
model.add(conv_base)  # 首先添加我们加载的VGG16模型
model.add(layers.Flatten())  # 添加全连接神经网络，即分类器
model.add(layers.Dense(units=512))
model.add(layers.Dropout(0.5))  # 使用Dropout，50%失活
model.add(layers.Dense(units=2, activation='softmax'))  # 最终分类为2

# 以文本显示模型超参数
model.summary()
# print(model.summary())

conv_base.trainable = False  # 第一步，设置网络模型不可训练,冻层操作

# 对训练数据使用数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# 验证数据不能使用模型增强
valid_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练数据导入
train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='categorical')
# 验证数据导入
validation_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='categorical')

# 对网络模型进行配置
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss=losses.binary_crossentropy,
              metrics=['acc'])

# Checkpoint
# 将训练好的模型进行保存
# model.save('VGG16_Base.h5')
path = "save_models"
if not os.path.exists(path):
    os.makedirs(path)
save_name = "model"
filepath = "save_models/vgg16_" + str(save_name) + "_-{epoch:03d}-{val_acc:.3f}.h5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
# 开始训练网络模型
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=1,  # 每一轮中的epoch，可以自己调
                              epochs=2,  # 进行20轮训练,可以自调
                              validation_data=validation_generator,
                              validation_steps=50,
                              callbacks=[checkpoint, ])

# 显示训练与验证过程的曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc:', history.history['acc'][-1])

json_data = {}
json_data['acc'] = acc
json_data['val_acc'] = val_acc
json_data['loss'] = loss
json_data['val_loss'] = val_loss

# 创建history_json文件夹将数据保存
path = "history_json"
if not os.path.exists(path):
    os.makedirs(path)

with open(f'history_json/vgg16_{save_name}.json', 'w') as f:
    json.dump(json_data, f)

# 重新训练上面训练好的模型
conv_base = model

# 解冻所有网络
conv_base.trainable = True

set_trainiable = False

for layer in conv_base.layers:
    if layer.name == 'block_conv1':
        set_trainiable = True
    if set_trainiable:
        layer.trainiable = True
    else:
        layer.trainiable = False

model = conv_base
# 配置网络
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss=losses.binary_crossentropy,
              metrics=['acc'])
# 进行训练
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=1,  # 每一轮中的epoch，可以自己调
                              epochs=2,  # 进行20轮训练,可以自调
                              validation_data=validation_generator,
                              validation_steps=50,
                              callbacks=[checkpoint, ])

# 显示训练与验证过程的曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc:', history.history['acc'][-1])

json_data = {}
json_data['acc'] = acc
json_data['val_acc'] = val_acc
json_data['loss'] = loss
json_data['val_loss'] = val_loss

with open(f'history_json/vgg16_finetune_{save_name}.json', 'w') as f:
    json.dump(json_data, f)

