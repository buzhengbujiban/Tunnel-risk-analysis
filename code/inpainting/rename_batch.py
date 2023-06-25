import os
path = "/data/MOCS2OPA/foreground/7/"
# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
#print(len(f))

n = 0
i = 0
for i in f:
    """
    # 设置旧文件名（就是路径+文件名）
    n=n+1
    if(n==9):
        break
    oldname = i
    print(i)
    ii=i.split("mask")[0]
    #print(ii)
    if(ii==""):
        continue
    try:
        oldinter=int(oldname.split(".jpg")[0])
    except:
        continue
    # 设置新文件名
    ss1 = "{0:0>12.0f}".format(oldinter)
    newname = str(oldinter) + ""
    # 用os模块中的rename方法对文件改名
    #os.rename(path+oldname, path+newname)
    print(oldname, '======>', newname)
"""
    if(i=="4871.jpg"):
        oldname=i
        newname="4871"
        print(oldname, '======>', newname)
        #os.rename(path+oldname, path+newname)
        break
