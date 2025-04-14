import pandas as pd
import matplotlib.pyplot as plt
import json
import math

def plot_overlay(input_data):
    data = pd.read_csv(input_data)
    working_data = data[data["layer"] == "global"]
    working_data = working_data[
        (
                (working_data["quant_type"] == "Q4_K_M") &
                (working_data["imat"] == False)
        ) | (
                (working_data["quant_type"] == "rate") &
                (working_data["dim"] == 3) &
                (working_data["threshold_low"] == 4.5) &
                (working_data["threshold_high"] == 4.5) &
                (working_data["imat"] == False)
        )
        # | (
        #         (working_data["quant_type"] == "rate") &
        #         (working_data["dim"] == 3) &
        #         (working_data["threshold_low"] == 4.5) &
        #         (working_data["threshold_high"] == 8.0) &
        #         (working_data["imat"] == True)
        # )
        ]

    # Get the histogram column (assuming two distributions)
    barplot_data = working_data["histogram"]
    cmap = plt.get_cmap('tab10')  # 'tab10' has 10 distinct colors
    plt.figure(figsize=(7, 4), dpi=250)


    for idx, data_str in enumerate(barplot_data):
        # Preprocess the string so that 'inf' is handled
        data_tmp = data_str.strip('"').replace("'", '"').replace("inf", "Infinity")
        data_list = json.loads(data_tmp)
        df = pd.DataFrame(data_list)

        row = working_data.iloc[idx]
        if str(row["quant_type"]) in ["rate","accu","prec"]:
            distribution_name = f'ZFP:{str(row["quant_type"])},bpw:{str(row["threshold_low"])},chunk:{str(4**row["dim"])}'
        else:
            distribution_name = str(row["quant_type"])

        # If any bin_end values are infinite, replace them with a virtual endpoint.
        if df['bin_end'].apply(math.isinf).any():
            # Find the maximum finite bin_end in the current histogram
            max_finite = df.loc[~df['bin_end'].apply(math.isinf), 'bin_end'].max()
            # Define an offset (adjust this value if needed)
            offset = 0
            # Replace infinite bin_end values with the computed virtual endpoint.
            df.loc[df['bin_end'].apply(math.isinf), 'bin_end'] = max_finite + offset

            # Calculate midpoints and bin widths
        df['bin_mid'] = (df['bin_start'] + df['bin_end']) / 2
        df['bin_width'] = df['bin_end'] - df['bin_start']
        color = cmap(idx % 10)
        plt.bar(df['bin_mid'], df['count'], width=df['bin_width'],
                edgecolor='black', align='center',
                color=color, alpha=0.6,
                label=distribution_name
                )

    plt.xlabel('Absolute difference')
    plt.ylabel('Count')
    plt.xlim([0, 0.010])
    plt.title('Elementwise difference to Llama-3.1-8B full precision model')
    plt.tight_layout()
    plt.legend(title="Quantization Kind")
    plt.savefig("Llama-3.1-8B-tensor_compare_Q4_K_M_ZFP_Rate4.50:4.50_3.pdf", bbox_inches='tight', pad_inches=0.05, transparent=True)



if __name__ == "__main__":
    input_data = "summary.csv"
    plot_overlay(input_data)