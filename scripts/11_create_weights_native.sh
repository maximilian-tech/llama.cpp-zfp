#!/bin/env bash
# run_create_default_weights.sh

set -euo pipefail


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/..

cd $ROOT_DIR

SOURCE_TYPE="F16"

if [[ "${1:-}" == "test" ]]; then
    models=( "3-8B" )
    imatrizes=( wi_imat no_imat )
    modes=( Q4_0 )

else
    models=( "3.1-8B" "3.1-70B" )
    imatrizes=( wi_imat no_imat )
    
    modes=( 
            Q4_0 
            Q4_1 
            Q5_0 
            Q5_1 
            IQ2_M 
            TQ1_0
            TQ2_0
            Q2_K
            Q2_K_S
            IQ3_XXS
            IQ3_S
            IQ3_M
            IQ3_XS
            Q3_K_S
            Q3_K_M
            Q3_K_L
            IQ4_NL
            IQ4_XS
            Q4_K_S
            Q4_K_M
            Q5_K_S
            Q5_K_M
            Q6_K
            Q8_0
            F16
            BF16
            IQ1_S
            IQ1_M
            IQ2_S
            IQ2_XXS
            IQ2_XS
          )
fi

for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
        for imatrix in "${imatrizes[@]}"; do

                    MODEL_SOURCE_DIR="${ROOT_DIR}/llm_unmodified_weights/Meta-Llama-${model}/"
                    MODEL_TARGET_DIR="${ROOT_DIR}/llm_experiment_weights/Meta-Llama-${model}/weights"
                    MODEL_TARGET_F16_DIR="${ROOT_DIR}/llm_experiment_weights/Meta-Llama-${model}/weights_F16"
                    
                    mkdir -p ${MODEL_TARGET_DIR} ${MODEL_TARGET_F16_DIR}
                    
                    MODEL_PREFIX="Meta-Llama-${model}"
                    
                    
                    if [[ "$imatrix" == "wi_imat" ]]; then
                        IMATRIX_APPEND="WII"
                    else
                        IMATRIX_APPEND="NOI"
                    fi
                    
                    RESULT_NAME="${MODEL_PREFIX}-${mode}+${IMATRIX_APPEND}"
                    RESULT_NAME_F16="${MODEL_PREFIX}-${SOURCE_TYPE}@${mode}+${IMATRIX_APPEND}"
                    
                    EXECUTABLE="${ROOT_DIR}/llama.cpp-cpu/bin/llama-quantize"
                    if [[ ! -x "$EXECUTABLE" ]]; then
                        echo "Error: Executable not found: $EXECUTABLE"
                        exit 1
                    fi

                    if [[ "$imatrix" == "wi_imat" ]]; then
                        IMATRIX_OPTION="--imatrix ${MODEL_SOURCE_DIR}/imatrix.dat"
                    else
                        IMATRIX_OPTION=""
                    fi


                    #mkdir -p ${ROOT_DIR}/{jobs,logs,result_quant,weights,weights_F16}
                    mkdir -p ${ROOT_DIR}/{job_scripts,job_logs,job_results}/${MODEL_PREFIX}/quantization
                    
                    
                    JOB_SCRIPT="${ROOT_DIR}/job_scripts/${MODEL_PREFIX}/quantization/${RESULT_NAME}.sbatch"
                    LOG_PATH="${ROOT_DIR}/job_logs/${MODEL_PREFIX}/quantization/${RESULT_NAME}.out"
                    RESULT_FILE="${ROOT_DIR}/job_results/${MODEL_PREFIX}/quantization/${RESULT_NAME}.out"

                    INPUT_WEIGHTS=${MODEL_SOURCE_DIR}/${MODEL_PREFIX}-${SOURCE_TYPE}.gguf

                    OUTPUT_WEIGHTS=${MODEL_TARGET_DIR}/${RESULT_NAME}.gguf
                    OUTPUT_WEIGHTS_F16=${MODEL_TARGET_F16_DIR}/${RESULT_NAME_F16}.gguf
                    
                    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --job-name=${RESULT_NAME}
#SBATCH --output=${LOG_PATH}
#SBATCH --error=${LOG_PATH}
#SBATCH --mem=40G
#SBATCH -A p_darwin
#SBATCH --time=05:00:00
#SBATCH --hint=multithread

cat \$0

module purge
source $ROOT_DIR/source_env.cpp_cpu

set -euo pipefail

time srun "$EXECUTABLE" \
    ${IMATRIX_OPTION} \
    "${INPUT_WEIGHTS}" \
    "${OUTPUT_WEIGHTS}" \
    ${mode} \
    \${SLURM_CPUS_PER_TASK} \
    | tee >( grep "^QUANT_RESULT" > "${RESULT_FILE}")

time srun "$EXECUTABLE" \
    --allow-requantize \
    "${OUTPUT_WEIGHTS}" \
    "${OUTPUT_WEIGHTS_F16}" \
    "${SOURCE_TYPE}" \
    \${SLURM_CPUS_PER_TASK}            

EOF

                    sync "$JOB_SCRIPT"

        done # imat
    done # model
done # mode
