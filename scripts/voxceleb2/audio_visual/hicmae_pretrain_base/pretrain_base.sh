model="pretrain_mutimae_dim64_512_patch4_160_a256"
OUTPUT_DIR="./saved/model/pretraining/voxceleb2/audio_visual/${model}_timemask_test"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p $OUTPUT_DIR
fi
# Set the path to pre-training dataset.
DATA_PATH='./saved/data/voxceleb2/info_clean_av_new.csv'
# batch_size can be adjusted according to number of GPUs
# this script is for 4 GPUs (1 nodes x 4 GPUs)

export MASTER_PORT=$((10000))
export OMP_NUM_THREADS=1

JOB_NAME=$hicmae_pretrain
#PARTITION=${PARTITION:-"video"}
PARTITION=${PARTITION:-"a6000"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

# batch_size can be adjusted according to the graphics card 
# --cpus-per-task 4\
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u test_run_mae_pretraining_av.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_type_audio time \
        --mask_ratio 0.9 \
        --input_size 160 \
        --mask_ratio_audio 0.8 \
        --input_size_audio 256 \
        --model pretrain_hicmae_dim512_patch16_160_a256 \
        --depths 16 \
        --batch_size 4 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 5 \
        --save_ckpt_freq 1 \
        --epochs 100 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 16 \
        --roll_mag_aug True \
        --return_intermediate_features True \
        --loss_weight 0.0025 \
        --inter_contrastive_temperature 0.07 \
        --use_frame_diff_as_target \
        --auto_resume \
        --start_epoch 1 \
        >${OUTPUT_DIR}/nohup1.out 2>&1 &
        

