model_paths=('../../gpt2baseline/gpt_baseline.pth'\
    '../../bartbaseline/bart_baseline.pth' '../../ours/ours_model.pth')
model_types=(gpt2 bart ours)
test_data_path='../../data/ROCStories_test.csv'

batch_size=128
temperature=0.8
mode="max"

MicroBleu(){
    CUDA_VISIBLE_DEVICES=$1
    if [ $CUDA_VISIBLE_DEVICES == 1 ];then
        CUDA_VISIBLE_DEVICES=5
    fi
    echo "Micro(sent) Handle ${model_types[$1]}, temperature = ${temperature}"
    python bleu.py --model ${model_paths[${1}]} --data $test_data_path\
        --batch_size $batch_size --device cuda --model-type ${model_types[${1}]}\
        --t ${temperature} --mode $2 --save_bleu 

}

Parallel(){
    for((i=0;i<3;i++))
    do
        (MicroBleu $i $mode)&
    done
    wait
}

Serial(){
    for((i=0;i<3;i++))
    do
        MicroBleu 0 $mode
    done
}

if [ -n "$1" ] && [ $1 = "parallel" ];then
    Parallel
elif [ -n "$1" ] && [ $1 = "serial" ];then
    Serial
else
    echo "Miss argument for bleu/run.sh: bash run.sh [parallel | serial]"
    exit
fi


