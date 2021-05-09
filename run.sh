trainGPT2(){
    cd ./gpt2baseline
    CUDA_VISIBLE_DEVICES=$1 bash train.sh
    cd ..
}

trainBart(){
    cd ./bartbaseline
    CUDA_VISIBLE_DEVICES=$1 bash train.sh
    cd ..
}

trainOurs(){
    cd ./ours
    CUDA_VISIBLE_DEVICES=$1 bash train.sh
    cd ..
}

trainSSC_Bert(){
    cd ./analyze/LSC_SSC/classification
    CUDA_VISIBLE_DEVICES=$1 bash train.sh
    cd ../../..
}

calculateBLEU(){
    cd ./analyze/bleu
    bash ./run.sh $1
    cd ../..
}

calculateLSC_SSC(){
    cd ./analyze/LSC_SSC
    bash ./run.sh
    cd ../..
}

Serial(){
    start_time=$(date +%s)
    echo "Training GPT-2 baseline"
    trainGPT2 0
    echo "Training Bart baseline"
    trainBart 0
    echo "Training Our model"
    trainOurs 0
    echo "Training SSC classifier (BERT)"
    trainSSC_Bert 0
    end_time=$(date +%s)
    elpased=$((${end_time}-${start_time}))
    return $elpased
}

Parallel(){
    start_time=$(date +%s)
    echo "Parallel Training GPT-2 baseline"
    (trainGPT2 2)&
    echo "Parallel Training Bart baseline"
    (trainBart 3)&
    echo "Parallel Training Our model"
    (trainOurs 4)&
    echo "Parallel Training SSC classifier (BERT)"
    (trainSSC_Bert 5)&
    wait
    end_time=$(date +%s)
    elpased=$((${end_time}-${start_time}))
    return $elpased
}

start_time=$(date +%s)
# We train models in parallel for efficiency, you can train them one by one
if [ -n "$1" ] && [ $1 = "parallel" ];then
    Parallel
    elpased=$?
    echo "Parallel training time = $(($elpased/60)) min $(($elpased%60)) sec" > parallel_traintime.txt
    echo "Start calculating BLEU"
    calculateBLEU "parallel"
elif [ -n "$1" ] && [ $1 = "serial" ];then
    Serial
    elpased=$?
    echo "Serial training time = $(($elpased/60)) min $(($elpased%60)) sec" > tee serial_traintime.txt
    echo "Start calculating BLEU"
    calculateBLEU "serial"
else
    echo "Miss argument: bash ./run.sh [parallel | serial]"
    exit
fi

echo "Start calculating LSC and SSC score"
calculateLSC_SSC
end_time=$(date +%s)
elpased=$((${end_time}-${start_time}))
echo "Exec time = $(($elpased/60)) min $(($elpased%60)) sec" > shell_exectime.txt
