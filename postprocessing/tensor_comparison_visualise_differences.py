import pandas as pd
import matplotlib.pyplot as plt
import json
import math
plt.style.use("default")
plt.rcParams.update({'figure.facecolor': 'white','axes.facecolor': 'white'})
plt.rc('font', family='serif')
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"]

patterns = ['///', '\\\\\\','/','\\']
def plot_overlay_multi(input_data):
    data = pd.read_csv(input_data)

    # Filter for the "global" layer
    working_data = data[data["layer"] == "global"]

    #----------------------------------------------------------------------
    # 1) DEFINE THE QUERIES FOR THE 4 SUBPLOTS
    # Adjust these filters based on your actual distributions.
    #----------------------------------------------------------------------
    queries = [
        {
            "name": "Quantization Error compared to F16 Llama-3.1-8B",
            "filter": (
                          (
                              (working_data["quant_type"] == "Q4_0")
                              |
                              ( (working_data["quant_type"] == "rate") &
                                (working_data["dim"] == 3) &
                                (working_data["threshold_low"] == 4.5) &
                                (working_data["threshold_high"] == 4.5)
                              )
                          ) &
                (working_data["imat"] == False)
            )
        },
        # {
        #     "name": "Rate=4.5:4.5, dim=3, no imat",
        #     "filter": (
        #                   (
        #                       (working_data["quant_type"] == "Q8_0")
        #                       |
        #                       ( (working_data["quant_type"] == "rate") &
        #                         (working_data["dim"] == 3) &
        #                         (working_data["threshold_low"] == 8.0) &
        #                         (working_data["threshold_high"] == 8.0)
        #                       )
        #                   ) &
        #         (working_data["imat"] == False)
        #     ),
        #     "xlim": [0,0.001]
        #
        # },
        # {
        #     "name": "Rate=4.5:8.0, dim=3, imat",
        #     "filter": (
        #         (working_data["quant_type"] == "rate") &
        #         (working_data["dim"] == 3) &
        #         (working_data["threshold_low"] == 4.5) &
        #         (working_data["threshold_high"] == 8.0) &
        #         (working_data["imat"] == True)
        #     )
        # },
        # {
        #     "name": "Example #4 (adjust me!)",
        #     "filter": (
        #         (working_data["quant_type"] == "hehe")  # or "prec" or some other quant_type
        #     )
        # }
    ]

    #----------------------------------------------------------------------
    # 2) CREATE A 2×2 FIGURE WITH 4 SUBPLOTS
    #----------------------------------------------------------------------
    #fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=250)
    fig, axes = plt.subplots(1, 1, figsize=(5.5, 3.8), dpi=250)
    try:
        axes = axes.ravel()  # Flatten the 2D array of axes for easier iteration
    except AttributeError as e:
        axes = [axes]
    # Disable scientific notation on the y-axis for all subplots.
    for ax in axes:
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        # Uncomment the next line if any offset text is still visible:
        # ax.get_yaxis().get_offset_text().set_visible(False)

    cmap = plt.get_cmap('tab10')

    #----------------------------------------------------------------------
    # 3) LOOP OVER THE 4 FILTERS AND PLOT THE NORMALIZED HISTOGRAM FOR EACH
    #----------------------------------------------------------------------
    for i, q in enumerate(queries):
        # Get the subset of data corresponding to the query
        sub_df = working_data[q["filter"]].copy()
        barplot_data = sub_df["histogram"]

        # Use the current subplot axis
        ax = axes[i]

        for idx, data_str in enumerate(barplot_data):
            # Preprocess the histogram data: adjust quotes and handle "inf"
            data_tmp = data_str.strip('"').replace("'", '"').replace("inf", "Infinity")
            data_list = json.loads(data_tmp)
            df = pd.DataFrame(data_list)

            # Replace infinite bin_end values with the maximum finite value in that row
            if df['bin_end'].apply(math.isinf).any():
                max_finite = df.loc[~df['bin_end'].apply(math.isinf), 'bin_end'].max()
                df.loc[df['bin_end'].apply(math.isinf), 'bin_end'] = max_finite

            # Compute bin midpoint and width
            df['bin_mid'] = (df['bin_start'] + df['bin_end']) / 2
            df['bin_width'] = df['bin_end'] - df['bin_start']

            # Compute the fraction per bin: count divided by total count
            total_count = df['count'].sum()
            if total_count > 0:
                df['fraction'] = df['count'] / total_count
            else:
                df['fraction'] = 0

            # Create a label for this histogram (only use it for the first entry)
            row = sub_df.iloc[idx]
            if str(row["quant_type"]) in ["rate", "accu", "prec"]:
                distribution_name = f'{row["quant_type"].title()}:{row["threshold_low"]}-Block:{int(4**row["dim"])}'
            else:
                distribution_name = str(row["quant_type"])

            color = cmap(idx % 10)
            ax.bar(df['bin_mid'], df['fraction'], width=df['bin_width'],
                   edgecolor='black', align='center',
                   color=color,
                   alpha=0.7,
                   label=distribution_name, #if idx == 0 else None
                   hatch=patterns[idx]
                   )

        # Set subplot title and labels.
        ax.set_title(q["name"])
        ax.set_xlabel("Absolute difference")
        ax.set_ylabel("Fraction")
        ax.set_xlim(q.get("xlim",[0, 0.01]))
        if not sub_df.empty:
            ax.legend(title="Quantization Type")

    plt.tight_layout()
    plt.savefig("Llama-3.1-8B-tensor_compare_Q4_K_M_ZFP_Rate4.50:4.50_3.pdf", bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.show()

if __name__ == "__main__":
    input_data = "summary.csv"
    plot_overlay_multi(input_data)
    print("Done!")
