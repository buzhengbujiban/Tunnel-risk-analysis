import argparse
import os
import re
import pandas as pd
from pandas import DataFrame
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled2.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled2.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.listdir(args.folder_path + "/training")
    validation_dirs = os.listdir(args.folder_path + "/validation")

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []
    cpp_file_names=[]


    # append all files into 2 lists
    for training_dir in training_dirs:
        # append each file into the list file names
        training_folder = os.listdir(args.folder_path + "/training" + "/" + training_dir)
        for training_item in training_folder:
            # modify to full path -> directory
            training_item = args.folder_path + "/training" + "/" + training_dir + "/"  +"maskk"+"/"
            training_file_names.append(training_item)
            x=training_dir
            zx=re.sub('[folder]', '', x)
            zzx=int(zx)
            ss1 = "{0:0>7.0f}".format(zzx)
            cpp_v_item= "com_pic/image/"+ ss1 + '.jpg'
            #print(cpp_v_item)
            cpp_file_names.append(cpp_v_item)
            #print(cpp_file_names)

    for j in range(len(cpp_file_names)):
        #mili="cp "+cpp_file_names[j]+" "+training_file_names[j]
        print(cpp_file_names[j])

        #os.system(mili)

        #print(mili)
    '''
    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)
'''