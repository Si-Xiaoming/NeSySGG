#!/bin/bash

#IMAGE_PATH=/home/odin/PycharmProjects/RelTR/data/vg/images/VG_100K/12.jpg
IMAGE_PATH=/home/odin/PycharmProjects/RelTR/data/vg/images/VG_100K/16.jpg
#IMAGE_PATH=/home/odin/PycharmProjects/RelTR/data/vg/images/VG_100K/626.jpg

python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/checkpoint0199.pth
#python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/nesy/pred_VG_constr_final_pair/checkpoint.pth
python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/nesy/pred_VG_constr_final_pair_trained/checkpoint.pth
#python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/nesy/pred_VG_constr_final_pair_trained_ce/checkpoint.pth
#python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/nesy/pred_VG_constr_final_triplet/checkpoint.pth
python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/nesy/pred_VG_constr_final_triplet_trained/checkpoint.pth
#python inference.py --img_path $IMAGE_PATH --resume /home/odin/PycharmProjects/RelTR/ckpt/nesy/pred_VG_constr_final_triplet_trained_ce/checkpoint.pth