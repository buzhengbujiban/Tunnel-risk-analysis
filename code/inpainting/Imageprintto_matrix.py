from PIL import Image
import pickle
import cv2
import numpy as np
im1 = Image.open("/home/lws/after_train_test/qj_output/output_0000001.jpg")
pixels1 = list(im1.getdata())
#print(pixels1)
'''
array = np.zeros([im1.size[0], im1.size[1], 4], dtype=np.uint8)
for x in range(im1.size[0]):
    for y in range(im1.size[1]):
        array[y, x, 3] = 0
img = Image.fromarray(array)
img.save('testrgba.png')
img = Image.open("/home/lws/testrgba.png")
pixels = np.array(img.getdata())
with open("matrixximg_cu0.txt","a+") as f:
    f.write(str(pixels))
print(type(im1.size))

'''



for i in range(len(pixels1)):
    if pixels1[i]==(174, 136, 61):
        pixels1[i]=(255,255,255)

#print(pixels1)

array = np.array(pixels1)
new_im = Image.fromarray(array)
image.save("/home/lws/after_train_test/qj_output2")
#with open("matrixximg_out.txt","a+") as f:
#    f.write(str(pixels))

#print(array)
arr=np.uint8(array)
arr=arr[np.newaxis,:]
#print(arr)
#cv2.imwrite('dst.png', arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


'''
with open("arr_dump.pickle", "wb") as f_out:
    pickle.dump(arr, f_out)

with open("arr_dump.pickle", "rb") as f_in:
    arr_new = pickle.load(f_in)
    print(arr_new.shape)
'''