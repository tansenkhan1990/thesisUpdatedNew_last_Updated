#dims=(10 100 1000)
#gamma=(1 10 20 30 40 50)
#temperature=1
#lrs=(0.01 0.05 0.1)
#batch_sizes=(1024)
#negs=(10 100 1000)
dims=(32 100)
gamma=(1 5 20)
temperature = 0
lrs=(0.00002 0.000002)
batch_sizes=(512)
negs=(10 500)
models=("ComplEx_quad")
regularization=(0 0.00001 0.0005 0.5 5)
#models=("TransE")
dataset="yago3_10/mapped"
train_with_groundings="false"
plot="false"
max_steps=200

CODE_PATH=".."
DATA_PATH=$dataset
#LOSS_FUNC=("margin_ranking")
LOSS_FUNC=("rotate")

executed_flag="false"
email_address="tansenkhan1990@gmail.com"
job_name="hyperparam_search"


# Always run the following line in your hpc user before using srun command
# source /home/sava096c/envs/env01/bin/activate
echo "starting grid run on all variables"

for model in "${models[@]}";do
for d in "${dims[@]}";do
  for g in "${gamma[@]}";do
    for lr in "${lrs[@]}";do
      for b in "${batch_sizes[@]}";do
        for neg in "${negs[@]}";do
          #for loss in "${LOSS_FUNC[@]}";do
            executed_flag="false"
#            while [ $executed_flag != "true" ];do
            for gpu_number in 0;do
                #SAVE_PATH="../models/$model/$loss/$dataset"
                #COMPLETE_SAVE_PATH="$SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/"
#               available_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i ${gpu_number})
#               extract the integer value in MB
#               available_mem=${available_mem//[^0-9]/}
#               echo "available mem is: $available_mem"
#               if [[ ${available_mem} -gt 1980 ]];then
                command="python train_version_2.py --name $model --dim $d --lr $lr --neg_sample $neg --gamma $g --temp $temperature --regul True --max_epoch $max_steps  --fifthopole True" --batch_size $b  --data_dir $DATA_PATH

                echo  "following command is executed"
                echo  $command >> ../hyperparameters_command/commands.txt
#                   executed_flag="true"
#                   sleep 15
#                   break
#               fi
#               sleep 5
#            done
done
done
done
done
done
done
done
done

wait
echo "all executed commands are finished"