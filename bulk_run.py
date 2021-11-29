#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>bulk_run_1.out 2>&1

a1=( 0 0 0 0 1 1 1 1 )
a2=( 0 0 1 1 0 0 1 1 )
a3=( 0 1 0 1 0 1 0 1 )

declare -i i=0
while [ "${a1[i]}" -a "${a3[i]}" ]; do
    echo 'gcn_single' ${a1[i]} 'bottom_att' ${a2[i]} 'gcn_att' ${a3[i]}
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
    --exp_name 'gcn_single'_${a1[i]}_${a2[i]}_${a3[i]} \
    --gpu_id 0 \
    --gcn_replace_att ${a1[i]} \
    --first_att ${a2[i]} \
    --second_att ${a3[i]} \
    --exp_name concatgcnattention \
    --embedding_mode word2vec \
    --embedding_file data/embeddings/word2vec_sg0_100.model \
    --attention_mode label \
    --d_a 512 \
    RNN  \
    --rnn_model LSTM \
    --n_layers 1 \
    --bidirectional 1 \
    --hidden_size 512 && nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9 && rm -rf checkpoints
      
    ((i++))

done
