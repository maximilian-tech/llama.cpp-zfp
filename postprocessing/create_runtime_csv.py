import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path
def parse_model(filename: str):
    """
    Meta-Llama-3.1-8B-F16_Q4_1_no_imat
    Meta-Llama-3.1-8B-F16_from_ZFP-acc_0.01-0.13_wi_imat_dim_1
    """

    modelname = Path(filename).stem

    model_name = modelname.strip().removeprefix("Meta-Llama-")

    llama_version = model_name.split("-", 1)[0]  # 3
    num_parameter = model_name.split("-", 2)[1]  # 8B
    try:
        processing_type, source_type = model_name.split("-", 2)[2].split("@", 1)
    except ValueError:
        processing_type =  source_type = model_name.split("-", 2)[2].split("@", 1)[0]

    if "ZFP" in source_type:
        source_type_tmp = source_type[3:]
        quant_type = source_type_tmp[:4]

        source_type_tmp2 = source_type_tmp[4:].split("_", 1)[0]
        threshold_low, threshold_high = source_type_tmp2.split(":")

        imat_tmp = source_type_tmp.split("_")[-1]
        dim = source_type_tmp.split("_")[-2]
    else:
        dim = None
        threshold_low = None
        threshold_high = None

        quant_type, imat_tmp = source_type.split("+")

    if imat_tmp == "NOI":
        imat = False
    elif imat_tmp == "WII":
        imat = True
    else:
        raise ValueError(f"Imhat has wrong value: {imat}")

    print(
        f"{model_name=} {llama_version=} {num_parameter=} {processing_type=} {quant_type=} {dim=} {threshold_low=} {threshold_high=} {imat=}")

    return {
        "model_name": model_name,
        "llama_version": llama_version,
        "num_parameter": num_parameter,
        "processing_type": processing_type,
        "quant_type": quant_type,
        "dim": dim,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "imat": imat
    }

def parse_info(text: str) -> dict:
    """
    Parses a given multiline string containing metadata and performance information into a dictionary.

    Expected text format:
        Meta-Llama-3.1-8B-Q4_0+NOI,ncores,24,iteration,1,node,n1310
        llama_perf_context_print: prompt eval time =    2789,96 ms /   154 tokens (   18,12 ms per token,    55,20 tokens per second)
        llama_perf_context_print:        eval time =   18947,69 ms /   199 runs   (   95,21 ms per token,    10,50 tokens per second)

    The output dict will have the following keys:
        - name
        - ncode            (extracted from the value paired with the "ncores" label)
        - iteration
        - node
        - prompt_eval_time [ms]
        - prompt_eval_throughput [token/s]
        - eval_time [ms]
        - eval_throughput [tokens/s]

    Returns:
        A dictionary with the parsed data.
    """
    # Split input text into non-empty lines
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    result = {}

    # Process the first line for metadata: name, ncode, iteration, node
    # The assumption: the first token is the name, then key/value pairs follow.
    # Example tokens: ["Meta-Llama-3.1-8B-Q4_0+NOI", "ncores", "24", "iteration", "1", "node", "n1310"]
    meta_tokens = lines[0].split(',')
    if len(meta_tokens) < 7:
        raise ValueError("Metadata line does not contain enough tokens")

    result["name"] = meta_tokens[0]
    # Here we ignore the labels (meta_tokens[1], meta_tokens[3], meta_tokens[5])
    # and map the values to the expected fields.
    try:
        # Convert to integer if possible.
        result["ncore"] = int(meta_tokens[2])
    except ValueError:
        result["ncore"] = meta_tokens[2]

    try:
        result["iteration"] = int(meta_tokens[4])
    except ValueError:
        result["iteration"] = meta_tokens[4]

    result["node"] = meta_tokens[6]

    # Define a helper to convert comma-decimal numbers to float.
    def parse_number(num_str: str) -> float:
        # Replace comma with dot and convert to float.
        return float(num_str.replace(',', '.'))

    # Process the prompt eval performance line (assumed to be the second non-empty line)
    if len(lines) > 1:
        prompt_line = lines[1]
        # Extract prompt eval time
        prompt_time_match = re.search(r'([\d,]+) ms per token', prompt_line)
        if prompt_time_match:
            result["prompt_eval_time"] = parse_number(prompt_time_match.group(1))
        else:
            result["prompt_eval_time"] = None

        # Extract prompt throughput (tokens per second)
        prompt_throughput_match = re.search(r'([\d,]+) tokens per second', prompt_line)
        if prompt_throughput_match:
            result["prompt_eval_throughput"] = parse_number(prompt_throughput_match.group(1))
        else:
            result["prompt_eval_throughput"] = None

    # Process the eval performance line (assumed to be the third non-empty line)
    if len(lines) > 2:
        eval_line = lines[2]
        # Extract eval time
        eval_time_match = re.search(r'([\d,]+) ms per token', eval_line)
        if eval_time_match:
            result["eval_time"] = parse_number(eval_time_match.group(1))
        else:
            result["eval_time"] = None

        # Extract eval throughput (tokens per second)
        eval_throughput_match = re.search(r'([\d,]+) tokens per second', eval_line)
        if eval_throughput_match:
            result["eval_throughput"] = parse_number(eval_throughput_match.group(1))
        else:
            result["eval_throughput"] = None

    return result

def parse_file_model_performance(filename):
    with open(filename,'r') as f:
        data = f.read()
        parsed_info = parse_info(data)

        filename_for_parser = "_".join(filename.split("_")[:-2])
        setup = parse_model(filename_for_parser + ".arbritraryEndingForStemOperation")

        return {**setup,**parsed_info}


def collect_runtime_info(pattern="*.out"):
    files = glob.glob(pattern)

    results = []
    for filename in files:
        try:
            data = parse_file_model_performance(filename)
            #data['filename'] = filename
            results.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results

if __name__ == "__main__":
    search_dir = "../job_results/Meta-Llama-3.1-8B/runtime_performance/*"

    data = collect_runtime_info(search_dir)
    df = pd.DataFrame(data)
    df.to_csv("runtimes.csv",index=False)
