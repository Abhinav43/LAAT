#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>bulk_run_2.out 2>&1


for domain in FocalLoss55 FocalLoss66 FocalLossMultiLabel1111 DiceLoss BinaryDiceLoss DiceLoss12 FocalLossV1 FocalLossV2 BPMLLLoss SoftDiceLossV2 SoftDiceLossV1 Dual_Focal_loss AsymmetricLoss AsymmetricLossOptimized
do
    for col_opss in none mean sum
    do
        printf 'running model' $domain 'with ops' $col_opss
        python3 -m src.run \
        --problem_name mimic-iii_2_50 \
        --max_seq_length 4000 \
        --n_epoch 50 \
        --patience 5 \
        --batch_size 8 \
        --optimiser adamw \
        --lr 0.001 \
        --dropout 0.3 \
        --level_projection_size 128 \
        --main_metric micro_f1 \
        --exp_name 'experiment'_$domain_$col_opss \
        --loss_name $domain \
        --reduction $col_opss \
        --embedding_mode word2vec \
        --embedding_file data/embeddings/word2vec_sg0_100.model \
        --attention_mode label \
        --d_a 512 \
        RNN  \
        --rnn_model LSTM \
        --n_layers 1 \
        --bidirectional 1 \
        --hidden_size 512 && nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9 && rm -rf checkpoints
    done
done