#!/bin/bash
# run_tensor_comparison.sh

set -euo pipefail
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/..
cd $ROOT_DIR

SOURCE_TYPE="F16"


EXECUTABLE_COMP="${ROOT_DIR}/llama.cpp-cpu/bin/llama-compare-tensors"

export NCPUS=1
SETTINGS="--histogram --per-layer-stats"


if [[ "${1:-}" == "test" ]]; then
    models=( "3.1-8B" )
else

    models=( "3.1-8B" "3.1-70B" )

fi


for model in "${models[@]}"; do

    MODEL_SOURCE_DIR="${ROOT_DIR}/llm_experiment_weights/Meta-Llama-${model}/weights_F16"
    MODEL_PREFIX="Meta-Llama-${model}"
    
    REFERENCE="${ROOT_DIR}/llm_unmodified_weights/Meta-Llama-${model}/${MODEL_PREFIX}-F16.gguf"
    
    
    RAM=$([[ "$model" =~ 70B ]] && echo "30G" || echo "20G")
    
    if [[ "${1:-}" == "test" ]]; then
        search_command="find ${MODEL_SOURCE_DIR} -type f -size +14G | head -n 1"
    else
        search_command="find ${MODEL_SOURCE_DIR} -type f -size +14G"
    fi

    echo "Search Command: '${search_command}'"
    
    for INPUT_WEIGHTS in $(eval "$search_command") ; do
        RESULT_NAME="$(basename -- "$INPUT_WEIGHTS" .gguf)"
        
        echo "Found Model: ${INPUT_WEIGHTS}"
        
        mkdir -p ${ROOT_DIR}/{job_scripts,job_logs,job_results}/${MODEL_PREFIX}/tensor_comparison
        
        JOB_SCRIPT="${ROOT_DIR}/job_scripts/${MODEL_PREFIX}/tensor_comparison/${RESULT_NAME}.sbatch"
        LOG_PATH="${ROOT_DIR}/job_logs/${MODEL_PREFIX}/tensor_comparison/${RESULT_NAME}.out"
        RESULT_FILE="${ROOT_DIR}/job_results/${MODEL_PREFIX}/tensor_comparison/${RESULT_NAME}.out"
        
        echo "Create Jobsscript for '${RESULT_NAME}'"
        
        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${NCPUS}
#SBATCH --mem=${RAM}
#SBATCH -A p_darwin
#SBATCH --job-name=${RESULT_NAME}
#SBATCH --output=${LOG_PATH}
#SBATCH --error=${LOG_PATH}
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread

module purge
source ${ROOT_DIR}/source_env.cpp_cpu

srun "${EXECUTABLE_COMP}" \
     ${SETTINGS} \
     --model-a "${REFERENCE}" \
     --model-b "${INPUT_WEIGHTS}" \
     2>&1 | tee ${RESULT_FILE}

EOF
        sync
        #sleep 0.05
        #sbatch "$JOB_SCRIPT"

    done # gguf-file
done # model 
