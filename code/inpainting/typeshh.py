import os
for i in range(10051,20051,1):
    ss1="mkdir -p ~/training_data/validation/val_folder"+str(i)+"\n"
    sss1="~/training_data/validation/val_folder"+str(i)+"\n"
    numm='{0:0=7.0f}'.format(i)
    ss2="mv "+"~/instances_train_folder/instances_train/instances_train/"+numm+".jpg "+sss1+"\n"
    print(ss2)
    with open("shh.txt","a+") as f:
        f.write(ss1)
        f.write(ss2)
    #os.system(ss1)
    #os.system(ss2)