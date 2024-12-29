server=170
pretrain_dataset='voxceleb2'
# dataset
finetune_dataset='dfew'
num_labels=7
# model
model_dir="pretrain_mutimae_dim64_512_patch4_160_a256_test"
output_model="pretrain_mutimae_dim_512_patch16_160_a256_a40_3"
ckpt=49
# input
input_size=160
input_size_audio=256
sr=4
# parameter
lr=1e-3
epochs=50

export MASTER_PORT=$((10000))
export OMP_NUM_THREADS=1

JOB_NAME=$hicmae_fintune
PARTITION=${PARTITION:-"a40"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

splits=(1, 2, 3, 4, 5)
for split in "${splits[@]}";
do
  # output directory
  OUTPUT_DIR="./saved/model/finetuning/${finetune_dataset}/audio_visual/${pretrain_dataset}_${output_model}/checkpoint-${ckpt}/eval_split0${split}_lr_${lr}_epoch_${epochs}_size${input_size}_a${input_size_audio}_sr${sr}_server${server}"
  if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
  fi
  # path to split files
  DATA_PATH="./saved/data/${finetune_dataset}/audio_visual//split0${split}"
  # path to pre-trained model
  MODEL_PATH="./saved/model/pretraining/${pretrain_dataset}/audio_visual/${model_dir}/checkpoint-${ckpt}.pth"

  srun -p $PARTITION \
      --job-name=${JOB_NAME} \
      --gres=gpu:${GPUS_PER_NODE} \
      --ntasks=${GPUS} \
      --ntasks-per-node=${GPUS_PER_NODE} \
      --kill-on-bad-exit=1 \
      ${SRUN_ARGS} \
      python -u test_run_class_finetuning_av.py \
      --model avit_dim512_patch16_160_a256 \
      --data_set ${finetune_dataset^^} \
      --nb_classes ${num_labels} \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 6 \
      --num_sample 1 \
      --input_size ${input_size} \
      --input_size_audio ${input_size_audio} \
      --short_side_size ${input_size} \
      --depths 16 \
      --save_ckpt_freq 1000 \
      --num_frames 16 \
      --sampling_rate ${sr} \
      --opt adamw \
      --lr ${lr} \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs ${epochs} \
      --dist_eval \
      --test_num_segment 2 \
      --test_num_crop 2 \
      --num_workers 16 \
      >${OUTPUT_DIR}/nohup.out 2>&1
done
echo "Done!"

