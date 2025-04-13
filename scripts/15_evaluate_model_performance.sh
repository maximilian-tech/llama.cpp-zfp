#!/bin/bash
# run_perplexity_eval.sh

set -euo pipefail


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/..
DATASET_DIR=${ROOT_DIR}/datasets
cd $ROOT_DIR

SOURCE_TYPE="F16"


EXECUTABLE_PPL="${ROOT_DIR}/llama.cpp-gpu/bin/llama-perplexity"
EXECUTABLE_CLI="${ROOT_DIR}/llama.cpp-gpu/bin/llama-cli"

HELLASWAG_NTASK=4000

CLI_PROMPT="How much wood would a woodchuck chuck if a woodchuck could chuck wood?"

export NCPUS=8
SETTINGS="-ngl 300 -s 1 -t ${NCPUS} --ctx-size 4096 "


if [[ "${1:-}" == "test" ]]; then
    models=( "3.1-8B" )
else
    #models=( "3-8B" "3-70B" "3.1-8B" "3.1-70B" )
    models=( "3.1-8B" "3.1-70B" )
    #models=( "3-8B" "3.1-8B" )
fi


for model in "${models[@]}"; do

    MODEL_SOURCE_DIR="${ROOT_DIR}/llm_experiment_weights/Meta-Llama-${model}/weights_F16"
    MODEL_PREFIX="Meta-Llama-${model}"
    
    NGPUS=$([[ "$model" =~ 70B ]] && echo "2" || echo "1")
    
    if [[ "${1:-}" == "test" ]]; then
        search_command="find ${MODEL_SOURCE_DIR} -type f -size +14G | head -n 1"
    else
        search_command="find ${MODEL_SOURCE_DIR} -type f -size +14G"
    fi

    echo "Search Command: '${search_command}'"
    
    for INPUT_WEIGHTS in $(eval "$search_command") ; do
        
        RESULT_NAME="$(basename -- "$INPUT_WEIGHTS" .gguf)"
        
        
        mkdir -p ${ROOT_DIR}/{job_scripts,job_logs,job_results}/${MODEL_PREFIX}/model_performance
        
        JOB_SCRIPT="${ROOT_DIR}/job_scripts/${MODEL_PREFIX}/model_performance/${RESULT_NAME}.sbatch"
        LOG_PATH="${ROOT_DIR}/job_logs/${MODEL_PREFIX}/model_performance/${RESULT_NAME}.out"
        RESULT_FILE="${ROOT_DIR}/job_results/${MODEL_PREFIX}/model_performance/${RESULT_NAME}"
        
        echo "Create Jobsscript for '${RESULT_NAME}'"
        
        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${NCPUS}
#SBATCH --mem=185G
#SBATCH -A p_darwin
#SBATCH --job-name=${RESULT_NAME}
#SBATCH --output=${LOG_PATH}
#SBATCH --error=${LOG_PATH}
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:${NGPUS}


module purge
source ${ROOT_DIR}/source_env.cpp_gpu

srun "${EXECUTABLE_CLI}" \
     ${SETTINGS} \
     -m "${INPUT_WEIGHTS}" \
     --repeat_penalty 1.0 \
     --prompt "${CLI_PROMPT}" \
     --predict 200 \
     2>&1 | tee ${RESULT_FILE}.cli

srun "${EXECUTABLE_PPL}" \
     ${SETTINGS} \
     --hellaswag \
     -f "${DATASET_DIR}/hellaswag_val_full.txt" \
     --hellaswag-tasks ${HELLASWAG_NTASK} \
     -m "${INPUT_WEIGHTS}" \
     2>&1 | tee ${RESULT_FILE}.hellaswag

srun "${EXECUTABLE_PPL}" \
     ${SETTINGS} \
     --perplexity \
     --file "${DATASET_DIR}/wiki.train.raw" \
     -m "${INPUT_WEIGHTS}" \
     2>&1 | tee ${RESULT_FILE}.ppl

#sync ${MODEL_SOURCE_DIR}/result_model_performance/${RESULT_NAME}.cli
#sync ${MODEL_SOURCE_DIR}/result_model_performance/${RESULT_NAME}.hellaswag
#sync ${MODEL_SOURCE_DIR}/result_model_performance/${RESULT_NAME}.ppl

#PPL_RESULT=\$(grep 'Final estimate: PPL =' "${MODEL_SOURCE_DIR}/result_model_performance/${RESULT_NAME}.ppl" 2>/dev/null) || PPL_RESULT="N/A"
#HSWAG_RESULT=\$(grep -Po "(?<=^${HELLASWAG_NTASK}[[:space:]]).+" "${MODEL_SOURCE_DIR}/result_model_performance/${RESULT_NAME}.hellaswag" 2>/dev/null) || HSWAG_RESULT="N/A"

#echo "${RESULT_NAME} -- \${PPL_RESULT:-N/A} -- HellaSwag: Score = \${HSWAG_RESULT:-N/A}"

EOF
        sync
        #sleep 0.05
        #sbatch "$JOB_SCRIPT"

    done # gguf-file
done # model 
