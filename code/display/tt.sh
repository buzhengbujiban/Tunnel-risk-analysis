export MODEL=efficientnet-lite0
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/${MODEL}.tar.gz
tar zxf ${MODEL}.tar.gz
wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt
python eval_ckpt_main.py --model_name=$MODEL --ckpt_dir=$MODEL --example_img=panda.jpg --labels_map_file=labels_map.txt