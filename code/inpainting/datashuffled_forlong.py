import argparse
import os
import re
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='/home/lws/training_data/long_random', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled-long.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled-long.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.listdir(args.folder_path)

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []
    cpp_file_names=[]


    # append all files into 2 lists
    for training_dir in training_dirs:
        # append each file into the list file names
        #print(training_dir)
        with open(file="/home/lws/vision/CompositionalGAN/dataset/face_sunglasses/paired2.txt", mode="a+") as f:
            f.write("/home/lws/training_data/long_random/" +training_dir+"\n")

    '''
    for j in range(len(training_file_names)):
        mili="cp "+cpp_file_names[j]+" "+training_file_names[j]
        #print(cpp_file_names[j],training_file_names[j])
        os.system(mili)

        #print(mili)

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