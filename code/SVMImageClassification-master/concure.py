import cv2
img = cv2.imread("D:\\1112.png",0)
blurred = cv2.GaussianBlur(img,(11,11),0) #高斯矩阵的长与宽都是11，标准差为0
gaussImg = img - blurred
cv2.imshow("Image",gaussImg)
cv2.waitKey(0)
