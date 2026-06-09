args="$@"
for arg in "$@"; do
    eval "$arg"
done

gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
master_port=${master_port:-16605}
n_gpus=$(( $n_machines * $gpus_per_node ))

proj_name=${proj_name:-"hypenet-Stage3"}
exp_group=${exp_group:-"MixedData"}
model_name=${model_name:-"LA"}
model_config=${model_config:-"hypenet-5b-mix_old_selection"}
train_config=${train_config:-"stage3"}
comment=${comment:-""}
tok_path=${tok_path:-"/path/to/models/qwen3-4B"}
data_path=${data_path:-"HuggingFaceFW/fineweb-edu"}
math_dir=${math_dir:-"/path/to/data/ultramath"}
web_token_ratio=${web_token_ratio:-0.8}

# Points to the mix-data Stage 2 output
stage2_ckpt=${stage2_ckpt:-"ckpt_20000"}
stage2_run_name=${stage2_run_name:-"MixedData_stage2_${model_config}_stage2-5b_"}
path_stage2=${path_stage2:-"/path/to/results/hypenet-Stage2/${stage2_run_name}/${stage2_ckpt}"}

run_name="${exp_group}_stage3_${model_config}_${train_config}_WebToken${web_token_ratio}"

cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/multigpu_config.yaml"
cmd+=" --num_machines=$n_machines"
cmd+=" --num_processes=$n_gpus"
cmd+=" --machine_rank=$machine_rank"
cmd+=" --main_process_port=$master_port"
cmd+=" stage3.py"
cmd+=" --init_from=${path_stage2}"
cmd+=" --init_method=build_from_stage2"
cmd+=" --proj_name=${proj_name}"
cmd+=" --model_name=${model_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --tok_path=${tok_path}"
cmd+=" --output_dir=/path/to/results"
cmd+=" --stage=3"
cmd+=" --train_config=configs/training/mix-data/${train_config}.json"
cmd+=" --model_config=configs/model/hypenet/${model_config}.json"
cmd+=" --data_path=${data_path}"
cmd+=" --math_dir=${math_dir}"
cmd+=" --web_token_ratio=${web_token_ratio}"

echo "==== Final command ===="
echo "$cmd"
echo "======================="
eval "$cmd"

echo "Stage 3 (mix-data) completed."
