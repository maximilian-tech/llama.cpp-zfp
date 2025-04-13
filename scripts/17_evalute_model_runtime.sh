#!/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/..

cd $ROOT_DIR


#CLI_PROMPT="How much wood would a woodchuck chuck if a woodchuck could chuck wood?"

# Approx. 154 Tokens
CLI_PROMPT="How much wood would a woodchuck chuck if a woodchuck could chuck wood? This age-old tongue twister has puzzled many, but let’s explore it from multiple angles. Scientifically, a woodchuck (or groundhog) doesn’t actually chuck wood, but if it could, we might estimate its capabilities based on its burrowing behavior. \
            According to a study, a woodchuck moves roughly 700 pounds of dirt when digging a burrow. If we equate this to wood, we might assume a woodchuck could chuck a similar amount. However, the physics of woodchucking would depend on its bite force, jaw strength, and endurance. Could it sustain wood-chucking for long durations, or would it tire quickly?"


if [[ "${1:-}" == "test" ]]; then
    models=( "3.1-8B" )
    modes=(
        #ZFP_from_ZFP-rate_4.00-4.00_no_imat_dim_3
        Q8_0_no_imat
    )
    cores=( 96 )
else
    models=( "3.1-8B" )
    cores=( 24 48 72 96 )
    modes=(
        ZFPrate4.00:4.00_2_NOI
        ZFPrate4.00:4.00_3_NOI
        ZFPrate4.00:4.00_4_NOI
        ZFPrate6.00:6.00_2_NOI
        ZFPrate6.00:6.00_3_NOI
        ZFPrate6.00:6.00_4_NOI
        ZFPrate8.00:8.00_2_NOI
        ZFPrate8.00:8.00_3_NOI
        ZFPrate8.00:8.00_4_NOI
        Q4_0+NOI
        Q4_1+NOI
        Q4_K_M+NOI
        Q4_K_S+NOI
        Q6_K+NOI
        Q8_0+NOI
    )
fi


for model in "${models[@]}" ; do

    MODEL_SOURCE_DIR="${ROOT_DIR}/llm_experiment_weights/Meta-Llama-${model}/weights"
    MODEL_PREFIX="Meta-Llama-${model}"

    for INPUT_WEIGHTS in "${modes[@]}" ; do
        
        RESULT_NAME="${MODEL_PREFIX}-${INPUT_WEIGHTS}"
        
        GGUF_F16_FILE=${MODEL_SOURCE_DIR}/${RESULT_NAME}.gguf
        if [ ! -f ${GGUF_F16_FILE} ]; then
            echo "File ${GGUF_F16_FILE} not found!"
            exit 2
        fi

        mkdir -p ${ROOT_DIR}/{job_scripts,job_logs,job_results}/${MODEL_PREFIX}/runtime_performance
        
        #echo "GGUF_F16_FILE_ABBR=${GGUF_F16_FILE_ABBR}"
        
        if [[ "${RESULT_NAME}" =~ .*_2_.* ]]; then
            EXECUTABLE_CLI="${ROOT_DIR}/llama.cpp-cpu/bin/llama-cli.rate.no_imat.dim_2"
        elif [[ "${RESULT_NAME}" =~ .*_3_.* ]]; then
            EXECUTABLE_CLI="${ROOT_DIR}/llama.cpp-cpu/bin/llama-cli.rate.no_imat.dim_3"
        elif [[ "${RESULT_NAME}" =~ .*_4_.* ]]; then
            EXECUTABLE_CLI="${ROOT_DIR}/llama.cpp-cpu/bin/llama-cli.rate.no_imat.dim_4"
        else
            EXECUTABLE_CLI="${ROOT_DIR}/llama.cpp-cpu/bin/llama-cli"
        fi
        echo "Create for ${RESULT_NAME}"
        
        for NCPUS in "${cores[@]}" ; do
            for i in {1..5}; do
                #JOB_SCRIPT="${MODEL_SOURCEDIR}/jobs_eval_performance/job_script_${OUTPUT_NAME}_n${NCPUS}_i${i}.sh"
                JOB_SCRIPT="${ROOT_DIR}/job_scripts/${MODEL_PREFIX}/runtime_performance/${RESULT_NAME}_n${NCPUS}_i${i}.sbatch"
                LOG_PATH="${ROOT_DIR}/job_logs/${MODEL_PREFIX}/runtime_performance/${RESULT_NAME}_n${NCPUS}_i${i}.out"
                RESULT_FILE="${ROOT_DIR}/job_results/${MODEL_PREFIX}/runtime_performance/${RESULT_NAME}_n${NCPUS}_i${i}.out"
                
                cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 104
#SBATCH --mem=200G
#SBATCH -A p_darwin
#SBATCH --job-name=${RESULT_NAME}
#SBATCH --output=${LOG_PATH}
#SBATCH --error=${LOG_PATH}
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --constraint=no_monitoring
#SBATCH --cpu-freq=2000000

cat \$0

module purge
source ${ROOT_DIR}/source_env.cpp_cpu

export OMP_NUM_THREADS=${NCPUS}
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

temp_file=$(mktemp) || { echo "Failed to create temp file" >&2; exit 1; }

NODE_NAME=\$(srun hostname)
echo "Node: \${NODE_NAME}"

time srun --cpu-bind=cores -c 104 -- \
    "${EXECUTABLE_CLI}" \
    -s 1 \
    -t ${NCPUS} \
    --ctx-size 4096 \
    -m "${GGUF_F16_FILE}" \
    --repeat_penalty 1.0 \
    --prompt "${CLI_PROMPT}" \
    --predict 200 \
    --ignore-eos \
    --no-mmap \
    2>&1 | tee \${temp_file}
    
sync \${temp_file}

# Extract values using grep and awk
#prompt_eval_time=\$(grep "prompt eval time" "\$temp_file" | awk '{print \$16}' | tr -d '\n' )
#eval_time=\$(grep "eval time" "\$temp_file" | awk '{print \$15}' | tr -d '\n' )

# Output to CSV (append mode)
echo "${RESULT_NAME},ncores,${NCPUS},iteration,${i},node,\${NODE_NAME}" > "${RESULT_FILE}"
grep "eval time" "\$temp_file" >> "${RESULT_FILE}"

rm "\$temp_file" && echo "Temporary file deleted."

EOF
            sync
            #sleep 0.05
            #sbatch "$JOB_SCRIPT"
            done #iteration
        done # cores
    done # gguf-file
done # model 