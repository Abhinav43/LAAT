#!/bin/bash
exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>log_bunch.out 3>&1

check_result() {
            input="experiments_tracking.txt"
            while IFS= read -r line
            do
                if [ "$1" == "$line" ]; then
                    result=match
                    break
                 else
                    result=nomatch
                fi
            done < "$input"
            }




total=145
coun_exp=0

coun_exp=$((x++))
echo "current experiment ${coun_exp}";
echo "remain experiment $(($total-$coun_exp))"


#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    check_result "before_mat_RNN_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}"

    if [ "$result" == "match" ]; then
        echo "experiment already executed"
    else
        echo "before_mat_RNN_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}" >> experiments_tracking.txt

        python3 -m src.run \
        --problem_name mimic-iii_2_50\
        --max_seq_length 4000 \
        --n_epoch 50 \
        --patience 10 \
        --batch_size 8 \
        --optimiser adamw \
        --lr 0.001 \
        --dropout 0.3 \
        --gpu_id 0 \
        --gcn_drop 1 \
        --gcn_att  ${a111[jj]} \
        --concat_1_att ${a211[jj]} \
        --concat_2_att ${a311[jj]} \
        --concat_3_att ${a411[jj]} \
        --final_att    ${a511[jj]} \
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
    fi
    ((jj++))
done






total=145
coun_exp=0

coun_exp=$((x++))
echo "current experiment ${coun_exp}";
echo "remain experiment $(($total-$coun_exp))"

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    for domain in 0 1
    do  
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"
        
        check_result "before_mat_RNN_CNN_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_CNN_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
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
        fi
    done
    ((jj++))
done



#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 1 1 )
    a2=( 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a2[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_GCN_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_GCN_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
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
        
        fi

        ((i++))
    done
    ((jj++))
done






total=145
coun_exp=0

coun_exp=$((x++))
echo "current experiment ${coun_exp}";
echo "remain experiment $(($total-$coun_exp))"

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    for domain in 0 1
    do  
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_BIGRU_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_BIGRU_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_CNN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_BIGRU  \
                --rnn_att $domain
        fi
    done
    ((jj++))
done






total=145
coun_exp=0

coun_exp=$((x++))
echo "current experiment ${coun_exp}";
echo "remain experiment $(($total-$coun_exp))"

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    for domain in 0 1
    do  
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_CNN_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_CNN_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_CNN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_CNN_CON  \
                --cnn_filter_size 500 \
                --cnn_att $domain
        fi
    done
    ((jj++))
done








total=145
coun_exp=0

coun_exp=$((x++))
echo "current experiment ${coun_exp}";
echo "remain experiment $(($total-$coun_exp))"

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    for domain in 0 1
    do  
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_BIGRU_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_BIGRU_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_$domain" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_CNN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_BIGRU_CON  \
                --rnn_att $domain
        fi
    done
    ((jj++))
done


#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 1 1 )
    a2=( 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a2[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"
        check_result "before_mat_RNN_GCN_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_GCN_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_GCN_CON  \
                --gcn_att  ${a1[i]} \
                --gcn_drop  1 \
                --gcn_both ${a2[i]}
        fi
        ((i++))
    done
    ((jj++))
done

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 1 1 )
    a2=( 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a2[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_cnn_bigru_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_cnn_bigru_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_cnn_bigru_con  \
                --cnn_filter_size 500 \
                --cnn_att ${a1[i]} \
                --rnn_att ${a2[i]}
        fi

        ((i++))
    done
    ((jj++))
done



#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 0 0 1 1 1 1 )
    a2=( 0 0 1 1 0 0 1 1 )
    a3=( 0 1 0 1 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a3[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"


        check_result "before_mat_RNN_cnn_gcn_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_cnn_gcn_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}" >> experiments_tracking.txt


            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
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
        fi

        ((i++))
    done
    ((jj++))
done

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 0 0 1 1 1 1 )
    a2=( 0 0 1 1 0 0 1 1 )
    a3=( 0 1 0 1 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a3[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_BIGRU_GCN_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_BIGRU_GCN_CON_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}" >> experiments_tracking.txt
        
            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_BIGRU_GCN_CON  \
                --rnn_att  ${a1[i]} \
                --gcn_att  ${a2[i]} \
                --gcn_drop 1 \
                --gcn_both ${a3[i]}
        fi

        ((i++))
    done
    ((jj++))
done

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 1 1 )
    a2=( 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a2[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_rnn_cnn_bigru_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_rnn_cnn_bigru_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
                --embedding_mode word2vec \
                --embedding_file data/embeddings/word2vec_sg0_100.model \
                --attention_mode label \
                --d_a 512 \
                RNN_rnn_cnn_bigru_con  \
                --cnn_filter_size 500 \
                --cnn_att ${a1[i]} \
                --rnn_att ${a2[i]}
        fi

        ((i++))
    done
    ((jj++))
done


#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do
    a1=( 0 0 0 0 1 1 1 1 )
    a2=( 0 0 1 1 0 0 1 1 )
    a3=( 0 1 0 1 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a3[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"
        check_result "before_mat_RNN_rnn_cnn_gcn_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_rnn_cnn_gcn_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}" >> experiments_tracking.txt
        

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
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
        fi

        ((i++))
    done
    ((jj++))
done

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do

    a1=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
    a2=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
    a3=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
    a4=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a4[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"

        check_result "before_mat_RNN_cnn_gcn_bigru_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}_${a4[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_cnn_gcn_bigru_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}_${a4[i]}" >> experiments_tracking.txt

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
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
        fi

        ((i++))
    done
    ((jj++))
done

#!/bin/bash
a111=( 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 )
a211=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
a311=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
a411=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
a511=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )
declare -i jj=0

while [ "${a111[jj]}" -a "${a511[jj]}" ]; do

    a1=( 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 )
    a2=( 0 0 0 0 1 1 1 1 0 0 0 0 1 1 1 1 )
    a3=( 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 )
    a4=( 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 )

    declare -i i=0
    while [ "${a1[i]}" -a "${a4[i]}" ]; do
        coun_exp=$((x++))
        echo "current experiment ${coun_exp}";
        echo "remain experiment $(($total-$coun_exp))"
        check_result "before_mat_RNN_rnn_gcn_bigru__cnn_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}_${a4[i]}"

        if [ "$result" == "match" ]; then
            echo "experiment already executed"
        else
            echo "before_mat_RNN_rnn_gcn_bigru__cnn_con_${a111[jj]}_${a211[jj]}_${a311[jj]}_${a411[jj]}_${a511[jj]}_${a1[i]}_${a2[i]}_${a3[i]}_${a4[i]}" >> experiments_tracking.txt
        

            python3 -m src.run \
                --problem_name mimic-iii_2_50\
                --max_seq_length 4000 \
                --n_epoch 50 \
                --patience 10 \
                --batch_size 8 \
                --optimiser adamw \
                --lr 0.001 \
                --dropout 0.3 \
                --gcn_drop 1 \
                --gcn_att  ${a111[jj]} \
                --concat_1_att ${a211[jj]} \
                --concat_2_att ${a311[jj]} \
                --concat_3_att ${a411[jj]} \
                --final_att    ${a511[jj]} \
                --level_projection_size 128 \
                --main_metric micro_f1 \
                --exp_name experiment_base_attention_RNN_GCN \
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
        fi

        ((i++))
    done
    ((jj++))
done
