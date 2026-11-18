#!/bin/bash
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNDcyMGQ5Mi05YjAzLTRjN2QtYjQ0Yi1kOTM1ZDFiZWI3ODQifQ=="
export NEPTUNE_PROJECT="russo.alessandro199/SGG-VG"
export NEPTUNE_CUSTOM_RUN_ID=`date +"%Y%m%d%H%M%s%N" | md5sum`

# cd /home/pdimasi/sgg_av

# find port
export MASTER_PORT=16200

LOG_ELASTIC_DIR=log_torchrun
dataset=vg
img_folder=/home/odin/dimasi_thesis/sgg_av/data/vg/images/VG_100K/
ann_path=/home/odin/dimasi_thesis/sgg_av/data/vg/
batch=4
num_gpu=1
num_workers=2

# activate conda
eval "$(conda shell.bash hook)"
conda activate reltr
exec
# remove old file from neptune dir to save space
# neptune clear
output_dir=/home/odin/dimasi_thesis/ckpt/nesy/pred_VG_constr
mkdir -p $output_dir
output_log=/home/odin/PycharmProjects/LTN_SGG_VG/LTN_orig_not_pair_hier_only_3_and_5.txt
echo $output_dir

NESY_START_EPOCH=0
torchrun --nproc_per_node=$num_gpu --nnodes=1 --master_port $MASTER_PORT  --log_dir $LOG_ELASTIC_DIR \
main.py --dataset $dataset --img_folder $img_folder --ann_path $ann_path --batch_size $batch --num_workers $num_workers  \
               --epochs 125 --save_freq 10  --neg_constr_path constraint/prompted_positives_vg_final_neg_full.csv \
               --pos_constr_path constraint/prompted_positives_vg_final_pos_full.csv --enable_ltn True \
          --sat_loss_coef 0.1 --nms_sat_enable --rel_operator 1 --univ_step 2 --p_epoch 50 --rc_enable \
      --univ_p 6 --use_neptune False --resume /home/odin/PycharmProjects/RelTR/ckpt/checkpoint0074.pth --use_hierarchy --constr_2_off --constr_4_off --output_dir $output_dir > $output_log

