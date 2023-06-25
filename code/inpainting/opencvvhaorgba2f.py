import cv2
import numpy as np
for xx in range(1000):
    ss1="{0:0>7.0f}".format(xx+1)
    ss='/home/lzd/mask_black/'+ss1+'.jpg'
    image_path = ss
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #img[63,63,:]=[255,0,0,128]
    nn=np.zeros([img.shape[0],img.shape[1],4],dtype=int)
    print(nn)
    #with open("matrixximg_in.txt","w") as f:
    #    f.write(str(list(img)))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(list(img[i][j])==[0,0,0]):
                nn[i][j] = np.array([255,255,255,255])
            else:
                nn[i][j] = np.array([0,0,0,0])

    #with open("matrixximg_out.txt","w") as f:
    #    f.write(str(list(nn)))

    #print(arrh)
    #img[:,np.newaxis]

    #print(arr,arr1,arr2)
    cv2.imwrite("after_train_test/mask_fan_se/"+str(xx+1)+'qj.png', nn, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])