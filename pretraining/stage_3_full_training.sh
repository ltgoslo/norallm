#!/bin/bash

#SBATCH --job-name=11-normistral
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --partition=standard-g
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_XXXXXX
#SBATCH --output=logs/normistral-11b-%j.out
#SBATCH --error=logs/normistral-11b-%j.err

mkdir -p workdir
wd=$(realpath workdir)


# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

# singularity setup
CONTAINER="/scratch/project_465000144/pytorch-lumi_sles-rocm-5.5.1-python-3.10-pytorch-v2.0.1-apex-torchvision-torchdata-torchtext-torchaudio.sif"
# CONTAINER="/scratch/project_462000086/norwegian_gpt/Megatron-DeepSpeed/pytorch-lumi_sles-rocm-5.5.1-python-3.10-pytorch-v2.0.1-apex-torchvision-torchdata-torchtext-torchaudio.sif"
SING_BIND="/scratch/project_465000144,/flash/project_465000144"

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"

# hold separate logs for easier debugging
rm -rf separate-logs
mkdir -p separate-logs

LEARNING_RATE=1e-4

set -euo pipefail

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

LOAD_CHECKPOINT_PATH=/scratch/project_465000144/dasamuel/normistral/normistral-11b-masked
SAVE_CHECKPOINT_PATH=/scratch/project_465000144/dasamuel/normistral/normistral-11b-masked
TENSORBOARD_PATH="tensorboard/CUSTOMMODELNAME/$SLURM_JOB_ID"
# rm -rf "$SAVE_CHECKPOINT_PATH" "$TENSORBOARD_PATH" # Start from scratch

# Data
TRAIN_DATA_PATH="/scratch/project_465000144/dasamuel/normistral/train_data.txt"
VALID_DATA_PATH="/scratch/project_465000144/dasamuel/normistral/validation_data.txt"
TOKENIZER_PATH="/scratch/project_465000144/dasamuel/normistral/tokenizer"

PP_SIZE=2
TP_SIZE=2

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

# export MEMORY_OPT_ALLREDUCE_SIZE=2500000000
export MEMORY_OPT_ALLREDUCE_SIZE=1500000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

TRAIN_SAMPLES=61_440_000
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=4_096_000
LR_COOLDOWN_SAMPLES=40_960_000

NLAYERS=40
NHIDDEN=5120
NHEADS=32
HEAD_DIM=128
FFN_HIDDEN_SIZE=14336
SEQ_LEN=4096

SAVE_INTERVAL=1000
LOG_INTERVAL=10

EVAL_INTERVAL=1000
EVAL_STEPS=10

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr $LEARNING_RATE \
    --min-lr $LEARNING_RATE \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --lr-cooldown-samples $LR_COOLDOWN_SAMPLES \
    --clip-grad 2.5 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --hf-checkpoint /scratch/project_465000144/dasamuel/normistral/original_mistral_checkpoint/model.safetensors \
    --embedding-checkpoint /scratch/project_465000144/dasamuel/normistral/normistral-11b-pre/global_step1000 \
    --num_kv_attention_heads 8 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --attention-softmax-in-fp32 \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --kv-channels $HEAD_DIM \
    --seq-length $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type PretrainedFromHF\
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --init-method-std 0.0048 \
    --glu-activation swiglu \
    --no-bias-gelu-fusion
    --sync-tp-duplicated-parameters \
    --bf16 \
    --seed 42 \
    --position-embedding-type rotary \
    --make-vocab-size-divisible-by 128 \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --wandb_name "11B_normistral_masked" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1

mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$SLURM_JOB_ID.json"

cat <<EOF > $DS_CONFIG_PATH
{
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "gradient_clipping": 2.5,
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "bf16": {
        "enabled": true
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE \
    "

CMD=" \
    pretrain_mistral.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $SAVE_CHECKPOINT_PATH \
    --load $LOAD_CHECKPOINT_PATH \
    --train-weighted-split-paths-path $TRAIN_DATA_PATH \
    --valid-weighted-split-paths-path $VALID_DATA_PATH \
    --data-impl mmap \
    --dataloader-type single \
    --num-workers 4 \
     $DEEPSPEED_ARGS \
    "

# Bind masks from Samuel Antao
c=fe

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ ! -d $wd/cray-deps ] ; then
  rm -rf $wd/cray-deps
  mkdir $wd/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

srun \
    --label \
    --cpu-bind=mask_cpu:$BIND_MASK \
    singularity exec \
    -B /opt/cray:/opt/cray \
    -B $wd/cray-deps:/opt/cray-deps \
    -B $wd:/workdir \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ./launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)