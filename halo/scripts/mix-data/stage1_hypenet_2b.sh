args="$@"
for arg in "$@"; do
    eval "$arg"
done

proj_name=${proj_name:-"hypenet-Stage1"}
exp_group=${exp_group:-"MixedData"}
model_name=${model_name:-"LA"}
model_config=${model_config:-"hypenet-2b-mix"}
train_config=${train_config:-"stage1"}
comment=${comment:-""}
orig_model=${orig_model:-"/path/to/models/qwen3-1.7B"}

# Data sources (override via env var or key=value arg)
# data_path is the FineWeb-Edu source: either a HuggingFace Hub ID (streaming)
# or a local path previously saved with load_from_disk.
data_path=${data_path:-"HuggingFaceFW/fineweb-edu"}
math_dir=${math_dir:-"/path/to/data/ultramath"}
web_token_ratio=${web_token_ratio:-0.5}

run_name="${exp_group}_${model_config}_${train_config}_WebToken${web_token_ratio}"

gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
master_port=${master_port:-16603}
n_gpus=$(( $n_machines * $gpus_per_node ))

cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/multigpu_config.yaml"
cmd+=" --main_process_port=$master_port"
cmd+=" --machine_rank=$machine_rank"
cmd+=" --num_processes=$n_gpus"
cmd+=" --num_machines=$n_machines"
cmd+=" stage1.py"
cmd+=" --proj_name=${proj_name}"
cmd+=" --model_name=${model_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --train_config=configs/training/mix-data/${train_config}.json"
cmd+=" --model_config=configs/model/hypenet/${model_config}.json"
cmd+=" --init_from=${orig_model}"
cmd+=" --output_dir=/path/to/results"
cmd+=" --tok_path=${orig_model}"
cmd+=" --stage=1"
cmd+=" --data_path=${data_path}"
cmd+=" --math_dir=${math_dir}"
cmd+=" --web_token_ratio=${web_token_ratio}"

echo "==== Final command ===="
echo "$cmd"
echo "======================="
nvidia-smi
echo "======================="
eval "$cmd"

echo "Stage 1 completed."
