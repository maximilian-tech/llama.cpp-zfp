#!/bin/env bash
# run_create_zfp_weights.sh
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}/..

cd $ROOT_DIR

SOURCE_TYPE="F16"

if [[ "${1:-}" == "test" ]]; then
    models=( "3.1-8B" )
    imatrizes=( wi_imat no_imat )
    dims=( 3 )
    modes=( rate )
    
    rate_parameters=( 4.00  6.00 )
    prec_parameters=( 08 09 10 11 12 13 )
    accu_parameters=( 0.05 0.10 0.12 0.13 0.14 )
else
    # default
    models=( "3.1-8B" "3.1-70B" ) #"3-8B" "3-70B" 
    imatrizes=( wi_imat no_imat )
    dims=( 4 3 2 1 )
    modes=( rate prec accu )
    rate_parameters=( 3.00 3.50 4.00 4.50 5.00 6.00 8.00 )
    prec_parameters=( 05 06 07 08 09 10 ) # 08 ~ 6pbw
    accu_parameters=( 0.01 0.05 0.10 0.12 0.13 0.14 ) # 0.001 ~ 10bpw # 0.14 ~ 3bpw
fi


for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
        for imatrix in "${imatrizes[@]}"; do
            for DIM in "${dims[@]}"; do
                declare -n PARAMETERS="${mode}_parameters"
                
                for PARAMETER in "${PARAMETERS[@]}"; do
                    if [[ $mode == "rate" ]]; then
                        if [[ $imatrix == "wi_imat" ]]; then    
                            export ZFP_RATE_MIN=$(echo "$PARAMETER" | bc | awk '{printf "%.2f\n", $0}')
                            export ZFP_RATE_MAX=$(echo "8" | bc | awk '{printf "%.2f\n", $0}')
                        elif [[ $imatrix == "no_imat" ]]; then    
                            export ZFP_RATE=$PARAMETER
                            export ZFP_RATE_MIN=$ZFP_RATE
                            export ZFP_RATE_MAX=$ZFP_RATE
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        VALUE_MIN=$ZFP_RATE_MIN
                        VALUE_MAX=$ZFP_RATE_MAX
                    elif [[ $mode == "prec" ]]; then
                        if [[ $imatrix == "wi_imat" ]]; then    
                            
                            
                            continue # Do not create importance matrix values other than RATE
                            
                            
                            export ZFP_PREC_MIN=$(echo "$PARAMETER" | bc | awk '{printf "%02d\n", $0}')
                            export ZFP_PREC_MAX=$(echo "10" | bc | awk '{printf "%02d\n", $0}')
                        elif [[ $imatrix == "no_imat" ]]; then    
                            export ZFP_PREC=$PARAMETER
                            export ZFP_PREC_MIN=$ZFP_PREC
                            export ZFP_PREC_MAX=$ZFP_PREC
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        VALUE_MIN=$ZFP_PREC_MIN
                        VALUE_MAX=$ZFP_PREC_MAX
                    elif [[ $mode == "accu" ]]; then
                        
                        if [[ $imatrix == "wi_imat" ]]; then
                            
                            
                            continue # Do not create importance matrix values other than RATE
                            
                            
                            export ZFP_TOL_MIN=$(echo "0.01" | bc | awk '{printf "%.2f\n", $0}')
                            export ZFP_TOL_MAX=$(echo "$PARAMETER" | bc | awk '{printf "%.2f\n", $0}')
                        elif [[ $imatrix == "no_imat" ]]; then    
                            export ZFP_TOL=$PARAMETER
                            export ZFP_TOL_MIN=$ZFP_TOL
                            export ZFP_TOL_MAX=$ZFP_TOL
                        else 
                            echo "Unknown imatrix value '$imatrix'"; exit 1
                        fi
                        VALUE_MIN=$ZFP_TOL_MIN
                        VALUE_MAX=$ZFP_TOL_MAX
                    fi
                    
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
                    
                    RESULT_NAME="${MODEL_PREFIX}-ZFP${mode}${VALUE_MIN}:${VALUE_MAX}_${DIM}+${IMATRIX_APPEND}"
                    RESULT_NAME_F16="${MODEL_PREFIX}-${SOURCE_TYPE}@ZFP${mode}${VALUE_MIN}:${VALUE_MAX}_${DIM}+${IMATRIX_APPEND}"
                    
                    echo "Create for ${RESULT_NAME}"
                    
                    EXECUTABLE="${ROOT_DIR}/llama.cpp-cpu/bin/llama-quantize.${mode}.${imatrix}.dim_${DIM}"
                    if [[ ! -x "$EXECUTABLE" ]]; then
                        echo "Error: Executable not found: $EXECUTABLE"
                        exit 1
                    fi

                    if [[ "$imatrix" == "wi_imat" ]]; then
                        IMATRIX_OPTION="--imatrix ${MODEL_SOURCE_DIR}/imatrix.dat"
                    else
                        IMATRIX_OPTION=""
                    fi


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
#SBATCH -c 1
#SBATCH --job-name=${RESULT_NAME}
#SBATCH --output=${LOG_PATH}
#SBATCH --error=${LOG_PATH}
#SBATCH --mem=10G
#SBATCH -A p_darwin
#SBATCH --time=05:00:00
#SBATCH --hint=multithread

cat \$0

module purge
source ${ROOT_DIR}/source_env.cpp_cpu

set -euo pipefail

if [[ "$imatrix" == "wi_imat" ]]; then
    export ZFP_RATE_MIN=$VALUE_MIN
    export ZFP_RATE_MAX=$VALUE_MAX
    
    export ZFP_PREC_MIN=$VALUE_MIN
    export ZFP_PREC_MAX=$VALUE_MAX
    
    export ZFP_TOL_MIN=$VALUE_MIN
    export ZFP_TOL_MAX=$VALUE_MAX                            
else
    export ZFP_RATE=$VALUE_MIN
    
    export ZFP_PREC=$VALUE_MIN
    
    export ZFP_TOL=$VALUE_MAX
fi

time srun "$EXECUTABLE" \
    ${IMATRIX_OPTION} \
    "${INPUT_WEIGHTS}" \
    "${OUTPUT_WEIGHTS}" \
    ZFP \
    \${SLURM_CPUS_PER_TASK} \
    | tee >( grep "^ZFP_RESULT" > "${RESULT_FILE}")

time srun "$EXECUTABLE" \
    --allow-requantize \
    "${OUTPUT_WEIGHTS}" \
    "${OUTPUT_WEIGHTS_F16}" \
    "${SOURCE_TYPE}" \
    \${SLURM_CPUS_PER_TASK}    


EOF
                    sync "$JOB_SCRIPT"
                done # parameter
            done # dim 
        done # imat
    done # model
done # mode
