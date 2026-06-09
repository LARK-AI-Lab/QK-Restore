args="$@"
for arg in "$@"; do
    eval "$arg"
done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
master_port=${master_port:-16604}
n_gpus=$(( $n_machines * $gpus_per_node ))

proj_name=${proj_name:-"hypenet-Stage2"}
exp_group=${exp_group:-"MixedData"}
model_name=${model_name:-"LA"}
train_config=${train_config:-"stage2-9b"}
model_config=${model_config:-"hypenet-9b-mix_old_selection"}
orig_model=${orig_model:-"/path/to/models/qwen3-8B"}
comment=${comment:-""}
data_path=${data_path:-"HuggingFaceFW/fineweb-edu"}
math_dir=${math_dir:-"/path/to/data/ultramath"}
web_token_ratio=${web_token_ratio:-0.8}

stage1_run_name=${stage1_run_name:-"MixedData_hypenet-9b-mix_old_selection_stage1_WebToken${web_token_ratio}"}
stage1_ckpt=${stage1_ckpt:-"ckpt_20000"}

path_stage1=${path_stage1:-"/path/to/results/hypenet-Stage1/${stage1_run_name}/${stage1_ckpt}"}
run_name="${exp_group}_stage2_${model_config}_${train_config}_${comment}"

cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/ds.yaml"
cmd+=" --num_machines=$n_machines"
cmd+=" --num_processes=$n_gpus"
cmd+=" --machine_rank=$machine_rank"
cmd+=" --main_process_port=$master_port"
cmd+=" stage2.py"
cmd+=" --init_from=${path_stage1}"
cmd+=" --proj_name=${proj_name}"
cmd+=" --model_name=${model_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --tok_path=${orig_model}"
cmd+=" --teacher_model=${orig_model}"
cmd+=" --output_dir=/path/to/results"
cmd+=" --stage=2"
cmd+=" --train_config=configs/training/mix-data/${train_config}.json"
cmd+=" --model_config=configs/model/hypenet/${model_config}.json"
cmd+=" --data_path=${data_path}"
cmd+=" --math_dir=${math_dir}"
cmd+=" --web_token_ratio=${web_token_ratio}"

echo "==== Final command ===="
echo "$cmd"
echo "======================="
nvidia-smi
echo "======================="
eval "$cmd"

echo "Stage 2 (mix-data) completed."
