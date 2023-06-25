import cv2

image=cv2.imread("/data/TsinghuaMOCS/mask_train/0000009.png",cv2.IMREAD_UNCHANGED)
print(image.max())
