#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>bulk_run_1.out 2>&1


for domain in BCE FL CBloss R-BCE-Focal NTR-Focal DBloss-noFocal CBloss-ntr DBloss base FocalLoss11 FocalLoss22 FocalLoss44 FocalLoss33 DICE_LOSS11
do
    for col_opss in none mean sum
    do
        printf 'running model' $domain 'with ops' $col_opss
        python3 -m src.run \
        --problem_name mimic-iii_2_50 \
        --max_seq_length 4000 \
        --n_epoch 1 \
        --patience 5 \
        --batch_size 8 \
        --optimiser adamw \
        --lr 0.001 \
        --dropout 0.3 \
        --level_projection_size 128 \
        --main_metric micro_f1 \
        --exp_name 'gcn_single'_$domain_$col_opss \
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