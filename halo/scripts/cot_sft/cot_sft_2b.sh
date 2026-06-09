#!/bin/bash
# CoT-SFT training script using Accelerator + LMTrainer
# Follows the same pattern as stage1/2/3

args="$@"
for arg in "$@"; do
    eval "$arg"
done

# Default values
gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
master_port=${master_port:-6604}
n_gpus=$(( $n_machines * $gpus_per_node ))

proj_name=${proj_name:-"JetNemotron_CoT_SFT"} #MixedData_hypenet_CoT_SFT
exp_group=${exp_group:-"Jet-Nemotron-2b"}
run_name=${run_name:-"${exp_group}_Miro"}
comment=${comment:-""}
tok_path=${tok_path:-"/path/to/models/Jet-Nemotron-2B"}

# Model checkpoint to fine-tune from (e.g., after stage3)
init_from=${init_from:-"/path/to/models/Jet-Nemotron-2B"}
# init_from=${init_from:-"/path/to/models/qwen3-1.7B"}


cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/ds.yaml"
cmd+=" --num_machines=$n_machines"
cmd+=" --num_processes=$n_gpus"
cmd+=" --machine_rank=$machine_rank"
cmd+=" --main_process_port=$master_port"
cmd+=" stage_cot_sft.py"
cmd+=" --init_from=${init_from}"
cmd+=" --init_method=AutoModelForCausalLM.from_pretrained"
cmd+=" --proj_name=${proj_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --tok_path=${tok_path}"
cmd+=" --train_config=configs/training/cot_sft.json"
cmd+=" --output_dir=/path/to/results"

echo "==== Final command ===="
echo "$cmd"
echo "======================="
eval "$cmd"

echo "CoT-SFT training completed."
