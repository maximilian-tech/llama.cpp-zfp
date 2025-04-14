import glob

import pandas as pd
import glob
import os
from pathlib import Path
# Patterns for capturing layer statistics
import re
import math

layer_pattern = re.compile(
    r'(\S+)\s+: rmse ([\d\.eE+-]+), maxerr ([\d\.eE+-]+), 95pct<([\d\.eE+-]+), median<([\d\.eE+-]+)')
histogram_bin_pattern = re.compile(r'\[([\d\.]+), ([\d\.inf]+)\):\s+(\d+)')


def parse_filename(filename):
    stem = Path(filename).stem
    parts = stem.split('_')

    model = parts[0]
    idx = 1

    compression_type = None
    param_min = None
    param_max = None

    if parts[idx] == 'from':
        compression_type = parts[idx + 1].split('-')[1]
        param_min, param_max = parts[idx + 2].split('-')
        idx += 3

    quantization = parts[idx]
    idx += 1

    imat = parts[idx]
    idx += 1

    dim = None
    if idx < len(parts) and parts[idx].startswith('dim'):
        dim = parts[idx].split('dim')[1]

    return {
        'model': model,
        'compression_type': compression_type,
        'param_min': param_min,
        'param_max': param_max,
        'quantization': quantization,
        'imat': imat,
        'dim': dim
    }


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

def parse_file_tf_difference(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    setup = parse_model(filename)

    layers_data = []
    current_layer = None

    for line in lines:
        layer_match = layer_pattern.match(line)
        if layer_match:
            current_layer = {
                'layer': layer_match.group(1),
                'rmse': float(layer_match.group(2)),
                'maxerr': float(layer_match.group(3)),
                '95pct': float(layer_match.group(4)),
                'median': float(layer_match.group(5)),
            }
            layers_data.append(current_layer)
    return {**setup, 'layers': layers_data}

def process_all_files_tf_difference(pattern="*.out"):
    files = glob.glob(pattern)
    results = []
    for filename in files:
        try:
            data = parse_file_tf_difference(filename)
            data['filename'] = filename
            results.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results

def parse_file_quantization(filename):
    def convert_value(s):
        """
        Convert a string to int, float, bool, or None if possible.
        """
        s = s.strip()
        if s.upper() == "N/A":
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        # Try integer conversion if no decimal point is present
        try:
            if '.' in s:
                raise ValueError
            return int(s)
        except ValueError:
            # Fall back to float conversion
            try:
                return float(s)
            except ValueError:
                # Otherwise, return as a string
                return s

    def parse_line(line):
        """
        Parse a comma-separated line where the first field is the record type
        and the remaining fields are key-value pairs.
        """
        # Split the line and remove extra spaces
        fields = [f.strip() for f in line.split(",")]

        # The first field is the record identifier
        parsed = {"record": fields[0]}

        # The remaining fields must come in pairs (key, value)
        if (len(fields) - 1) % 2 != 0:
            raise ValueError("Expected key-value pairs; odd number of fields found after record identifier.")

        # Loop over key-value pairs
        for i in range(1, len(fields), 2):
            key = fields[i]
            value = fields[i + 1]
            parsed[key] = convert_value(value)

        return parsed



    with open(filename, 'r') as f:
        lines = f.readlines()

    setup = parse_model(filename)

    #layers_data = []
    current_layer = None
    # Parse the line
    parsed = parse_line(lines[0])

    # Extract the desired metrics
    n_elements = parsed.get("n_elements")
    bits_per_weight = parsed.get("bits_per_weight")
    size = parsed.get("compressed_size(MiB)")

    layers_data = {
        'n_elements': int(n_elements),
        'bits_per_weight': float(bits_per_weight),
        'size': size,
    }

    return {**setup, **layers_data}

def process_all_files_quantization(pattern="*.out"):
    files = glob.glob(pattern)
    results = []
    for filename in files:
        try:
            data = parse_file_quantization(filename)
            data['filename'] = filename
            results.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results

# ToDo. Extrace PPL + Hellaswag Values
def parse_file_model_performance(filename):

    def parse_ppl(lines: list[str]) -> float | None:
        for line in lines:
            if "- ETA" in line:
                try:
                    return_value = float(line.split("=")[1].split("+")[0])
                except:
                    return_value = None
                return return_value
        return None
    def parse_hellaswag(lines: list[str]) -> float | None :
        for line in lines:
            if line.startswith("4000"):
                return float(line.split()[-1])
        return None


    data={}
    endings = ["ppl", "hellaswag"]
    for end in endings:
        source_file = f"{filename}.{end}"
        with open(source_file, 'r') as f:
            lines = f.readlines()

            setup = parse_model(source_file)
            if source_file.endswith(".ppl"):
                value = parse_ppl(lines)
                data["ppl"] = value

            if source_file.endswith(".hellaswag"):
                value = parse_hellaswag(lines)
                data["hellaswag"] = value

    return {**setup, **data}
def process_all_files_model_performance(pattern="*.out"):
    files = glob.glob(pattern)

    files = [".".join(f.split(".")[:-1]) for f in files]
    files = sorted((set(files)))
    results = []
    for filename in files:
        try:
            data = parse_file_model_performance(filename)
            data['filename'] = filename
            results.append(data)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results

if __name__ == "__main__":
    # Example usage:
    if True:
        search_dir = "../job_results/Meta-Llama-3.1-*B/tensor_comparison/"
        # Example usage:
        data1 = process_all_files_tf_difference(f"{search_dir}*.out")

        # Convert to DataFrame for summary
        df_pointwise_difference = pd.json_normalize(data1, 'layers',
                               ['llama_version', 'num_parameter', 'processing_type', 'quant_type', 'dim', 'threshold_low',
                                'threshold_high', 'imat', 'model_name'], errors='ignore')
        df_pointwise_difference = df_pointwise_difference[df_pointwise_difference["layer"] == "global"]
        df_pointwise_difference["model_name"] = df_pointwise_difference["model_name"].str.replace("F16@", "", regex=False)

    if True:
        search_dir = "../job_results/Meta-Llama-3.1-*B/quantization/"
        data2 = process_all_files_quantization(f"{search_dir}*.out")
        df_quantization = pd.DataFrame(data2)


    if True:
        search_dir = "../job_results/Meta-Llama-3.1-*B/model_performance/*"
        data3 = process_all_files_model_performance(f"{search_dir}")
        df_model_performance = pd.DataFrame(data3)
        df_model_performance["model_name"] = df_model_performance["model_name"].str.replace("F16@", "",
                                                                                                  regex=False)


    merged_df1 = pd.merge(df_pointwise_difference, df_quantization[["model_name","n_elements","bits_per_weight","size"]], on="model_name", how="outer")
    merged_df2 = pd.merge(merged_df1, df_model_performance[["model_name","ppl","hellaswag"]], on="model_name", how="outer")
    # Save full summary to CSV
    merged_df2.to_csv("all_data.csv", index=False)

    # Create second DataFrame without histogram column
    # df_no_hist = df.drop(columns=['histogram'])
    # df_no_hist = df_no_hist[df_no_hist['layer'] == 'global']
    # print(df_no_hist)
    #
    # # Save second summary to CSV
    # df_no_hist.to_csv("summary_no_histogram.csv", index=False)

