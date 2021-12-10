# first

#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log_bunch.out 3>&1

total=145
coun_exp=0

coun_exp=$((x++))
echo "current experiment ${coun_exp}";
echo "remain experiment $(($total-$coun_exp))"


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        echo count_exp
        python3 -m src.run \
            --problem_name mimic-iii_2_50\
            --max_seq_length 4000 \
            --n_epoch 50 \
            --patience 10 \
            --batch_size 8 \
            --optimiser adamw \
            --lr 0.001 \
            --dropout 0.3 \
            --gpu_id 1 \
            --before_matmul_gcn_drop ${a111[ij]} \
            --before_matmul_gcn_att ${a211[ij]} \
            --before_matmul_mode $ops_s \
            --level_projection_size 128 \
            --main_metric micro_f1 \
            --exp_name experiment_base_attention \
            --embedding_mode word2vec \
            --embedding_file data/embeddings/word2vec_sg0_100.model \
            --attention_mode label \
            --d_a 512 \
            RNN  \
            --rnn_model LSTM \
            --n_layers 1 \
            --bidirectional 1 \
            --hidden_size 512
    done
    ((ij++))
done



#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        for domain in 0 1
        do  

            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_CNN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_CNN  \
                --cnn_filter_size 500 \
                --cnn_att $domain
        done
    done
    ((ij++))
done


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 1 1 )
        a2=( 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a2[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_GCN  \
                --gcn_att  ${a1[i]} \
                --gcn_drop  1 \
                --gcn_both ${a2[i]}

            ((i++))

        done
    done
    ((ij++))
done


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        for domain in 0 1
        do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_BIGRU \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_BIGRU  \
                --rnn_att $domain
        done
    done
    ((ij++))
done


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        for domain in 0 1
        do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_CNN_CON \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_CNN_CON  \
                --cnn_filter_size 500 \
                --cnn_att $domain
        done

    done
    ((ij++))
done


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        for domain in 0 1
        do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_BIGRU_CON \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_BIGRU_CON  \
                --rnn_att $domain
        done

    done
    ((ij++))
done


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 1 1 )
        a2=( 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a2[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN_CON \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_GCN_CON  \
                --gcn_att  ${a1[i]} \
                --gcn_drop 1 \
                --gcn_both ${a2[i]}

            ((i++))

        done
    done
    ((ij++))
done


#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 1 1 )
        a2=( 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a2[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_cnn_bigru_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_cnn_bigru_con  \
                --cnn_filter_size 500 \
                --cnn_att ${a1[i]} \
                --rnn_att ${a2[i]}

            ((i++))

        done
    done
    ((ij++))
done

#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 0 0 1 1 1 1 )
        a2=( 0 0 1 1 0 0 1 1 )
        a3=( 0 1 0 1 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a3[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_cnn_gcn_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_cnn_gcn_con  \
                --cnn_filter_size 500 \
                --cnn_att  ${a1[i]} \
                --gcn_att  ${a2[i]} \
                --gcn_drop 1 \
                --gcn_both ${a3[i]} 

            ((i++))

        done
    done
    ((ij++))
done

#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 0 0 1 1 1 1 )
        a2=( 0 0 1 1 0 0 1 1 )
        a3=( 0 1 0 1 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a3[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_cnn_gcn_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_BIGRU_GCN_CON  \
                --rnn_att  ${a1[i]} \
                --gcn_att  ${a2[i]} \
                --gcn_drop 1 \
                --gcn_both ${a3[i]} 

            ((i++))

        done
    done
    ((ij++))
done

#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 1 1 )
        a2=( 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a2[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_cnn_bigru_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_rnn_cnn_bigru_con  \
                --cnn_filter_size 500 \
                --cnn_att ${a1[i]} \
                --rnn_att ${a2[i]}

            ((i++))

        done
    done
    ((ij++))
done

#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 0 0 1 1 1 1 )
        a2=( 0 0 1 1 0 0 1 1 )
        a3=( 0 1 0 1 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a3[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_rnn_cnn_gcn_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_rnn_cnn_gcn_con  \
                --cnn_filter_size 500 \
                --cnn_att  ${a1[i]} \
                --gcn_att  ${a2[i]} \
                --gcn_drop 1 \
                --gcn_both ${a3[i]} 

            ((i++))

        done
    done
    ((ij++))
done

#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
        a2=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
        a3=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
        a4=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a4[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_cnn_gcn_bigru_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_cnn_gcn_bigru_con  \
                --cnn_filter_size 500 \
                --cnn_att  ${a1[i]} \
                --gcn_att  ${a2[i]} \
                --gcn_drop 1 \
                --gcn_both ${a3[i]} \
                --rnn_att  ${a4[i]}

            ((i++))

        done
    done
    ((ij++))
done

#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        a1=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
        a2=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
        a3=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
        a4=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )

        declare -i i=0
        while [ "${a1[i]}" -a "${a4[i]}" ]; do
            coun_exp=$((x++))
            echo "current experiment ${coun_exp}";
            echo "remain experiment $(($total-$coun_exp))"
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gpu_id 1 \
                --before_matmul_gcn_drop ${a111[ij]} \
                --before_matmul_gcn_att ${a211[ij]} \
                --before_matmul_mode $ops_s \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_rnn_gcn_bigru__cnn_con \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_rnn_gcn_bigru__cnn_con  \
                --cnn_filter_size 500 \
                --cnn_att  ${a1[i]} \
                --gcn_att  ${a2[i]} \
                --gcn_drop 1 \
                --gcn_both ${a3[i]} \
                --rnn_att  ${a4[i]}

            ((i++))

        done
    done
    ((ij++))
done
