#!/bin/bash
a111=( 0 0 1 1 )
a211=( 0 1 0 1 )
declare -i ij=0

while [ "${a111[ij]}" -a "${a211[ij]}" ]; do
    for ops_s in none cat sum catsum
    do
        echo $ops_s ${a111[ij]} ${a211[ij]}
    done
    ((ij++))
done


--before_matmul_gcn_drop ${a111[ij]} \
--before_matmul_gcn_att ${a211[ij]} \
--before_matmul_mode $ops_s \


import itertools
import pandas as pd

def gen_pos(how):
    rt = pd.DataFrame(list(itertools.product([0,1], repeat = how))).T.values.tolist()
    rt = [[str(m) for m in k] for k in rt]
    
    for i in range(len(rt)):
        print(" ".join(rt[i]))
        print('\n')
        
gen_pos(4)
