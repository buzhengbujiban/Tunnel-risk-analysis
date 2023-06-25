import os
import subprocess
import cv2
import numpy as np
os.system("pwd")
#os.system("conda activate lwss")
'''
import paramiko
#实例化ssh客户端`
ssh = paramiko.SSHClient()
#创建默认的白名单
policy = paramiko.AutoAddPolicy()
#设置白名单
ssh.set_missing_host_key_policy(policy)
#链接服务器
ssh.connect(
    hostname = "47.97.51.98", #服务器的ip
    port = 6001, #服务器的端口
    username = "lws", #服务器的用户名
    password = "lws2021" #用户名对应的密码
)
'''
for i in range(17,9000):
    xx=i
    ss1="{0:0>7.0f}".format(xx+1)
    maskss=' --mask /home/lzd/mask_black/'+ss1+'.jpg'
    imgss="python test.py --image /home/lws/instances_train_folder/instances_train/instances_train/"+ss1+'.jpg'
    restss=" --output /home/lws/after_train_test/output_lggaas/lggmaas_162892_in_py"+ss1+".jpg --checkpoint_dir lggmaas/full_model_celeba_hq_256"
    comm=imgss+maskss+restss
    '''
    # 远程执行命令
    stdin, stdout, stderr = ssh.exec_command("ls")
    # exec_command 返回的对象都是类文件对象
    # stdin 标准输入 用于向远程服务器提交参数，通常用write方法提交
    # stdout 标准输出 服务器执行命令成功，返回的结果  通常用read方法查看
    # stderr 标准错误 服务器执行命令错误返回的错误值  通常也用read方法
    # 查看结果，注意在Python3 字符串分为了：字符串和字节两种格式，文件返回的是字节
    result = stdout.read().decode()
    
    print(result)
    '''
    #print(comm)
    os.system(comm)