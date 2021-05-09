CUDA_VISIBLE_DEVICES=6
model_paths=('../../gpt2baseline/gpt_baseline.pth'\
    '../../bartbaseline/bart_baseline.pth' '../../ours/ours_model.pth')
model_types=(gpt2 bart ours)
test_data_path='../../data/ROCStories_test.csv'
output_path="test_output"

batch_size=128
temperature=0.8

Eval(){
    python evaluation.py --src $1 --output $2
}

for((i=0;i<3;i++))
do
    python generate.py --model ${model_paths[${i}]} --data ${test_data_path}\
		 --batch_size $batch_size --device cuda --output ${output_path}\
		 --model-type ${model_types[${i}]} -t $temperature

     Eval ${output_path}_${model_types[${i}]}_0.txt ${model_types[${i}]}_emotion_report.txt
     Eval ${output_path}_${model_types[${i}]}_1.txt ${model_types[${i}]}_event_report.txt
done
