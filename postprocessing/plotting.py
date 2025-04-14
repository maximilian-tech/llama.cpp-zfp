import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
#plt.rcParams["font.family"] = "Times New Roman"
plt.style.use("default")
plt.rcParams.update({'figure.facecolor': 'white','axes.facecolor': 'white'})
plt.rc('font', family='serif')
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"]
desired_order = ["accu", "prec", "rate", "built-in","native"]
# Example color, marker, and name mappings

color_mapping = {
    'rate': '#0173B2',    # Blue
    'accu': '#CC78BC',    # Pinkish
    'prec': '#009E73',    # Greenish
    'built-in': '#949494', # Gray
    'native': '#ffd92f' ,  # Yellow for native F16
    1:"blue",
    2:"red",
    3:"green",
    4:"orange",
}
marker_mapping = {
    'rate': 'o',      # Circle
    'accu': 's',      # Square
    'prec': 'D',      # Diamond
    'built-in': '^',   # Triangle
    'native': 'v',
    1: 'o',      # Circle
    2: 's',      # Square
    3: 'D',      # Diamond
    4: '^',   # Triangle

}
name_mapping = {
    'rate': 'ZFP-Rate',
    'accu': 'ZFP-Accuracy',
    'prec': 'ZFP-Precision',
    'built-in': 'Built-in Quantization',
    'native': 'F16 Native'
}

label_dict = {
    "ppl": "Perplexity (n_ctx=4096,  WikiText-2)",
    "hellaswag": "HellaSwag Score (higher is better)",
    "bpw":"Bits per weight",
    "compressed_size": "Compressed Model size (GiB)",
}


def plot_summary_ppl(df):

    my_df = df.copy()
    conditions = [
        my_df['quant_type'] == 'F16',
        my_df['quant_type'].isin(['rate', 'accu', 'prec'])
    ]
    choices = ['native', my_df['quant_type']]
    my_df['color_group'] = np.select(conditions, choices, default='built-in')

    my_df["size"] = my_df["size"] / 1024
    filtered_df = my_df[(my_df["ppl"] < 14)
                        & (my_df["quant_type"] != "BF16")
                        #& (my_df["imat"] == False)

    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=250)
    ax.set_xlabel("Llama-3.1 Compressed Model Size [GiB]")
    ax.set_ylabel(label_dict["ppl"])
    ax.set_title("Perplexity of Llama-3.1-8B/-70B")
    ax.tick_params(axis='both', which='major')

    for group in desired_order:
        # Check if the group is present in the filtered data
        if group not in filtered_df['color_group'].unique():
            continue  # skip if the group doesn't exist

        group_data = filtered_df[filtered_df["color_group"] == group]

        ax.scatter(
            group_data["size"],
            group_data["ppl"],
            color=color_mapping.get(group, 'black'),
            marker=marker_mapping.get(group, 'o'),
            label=name_mapping.get(group, "unknown"),
            edgecolors='black',
            s=65,
            alpha=0.9
        )

    ax.set_xscale('log')

    # Disable scientific notation on the x-axis
    ax.set_xticks([2,5,10,20,50,100])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(title="Quantization Type", loc='upper right')
    plt.tight_layout()
    #plt.show()
    plt.savefig("overview_8B_70B.pdf",transparent=True,dpi=300)
    plt.close()

def plot_zfp_8b(df):

    my_df = df.copy()
    conditions = [
        my_df['quant_type'] == 'F16',
        my_df['quant_type'].isin(['rate', 'accu', 'prec'])
    ]
    choices = ['native', my_df['quant_type']]
    my_df['color_group'] = np.select(conditions, choices, default='built-in')

    my_df["size"] = my_df["size"] / 1024
    filtered_df = my_df[(my_df["ppl"] < 8)
                        & (my_df["quant_type"] != "BF16")
                        & (
                                my_df["quant_type"].str.startswith("Q")
                             | my_df["quant_type"].str.startswith("rate")
                             | my_df["quant_type"].str.startswith("prec")
                             | my_df["quant_type"].str.startswith("accu")
                             | ("I" not in my_df["quant_type"])
                        )
                        #& (my_df["color_group"] != "built-in")
                        & (my_df["num_parameter"] == "8B")
                        & (my_df["imat"] == False)
                        & (my_df["size"] < 10)
    ]
    print(my_df)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=250)
    ax.set_xlabel(label_dict["bpw"])
    ax.set_ylabel(label_dict["ppl"])
    ax.set_title("Perplexity of Llama-3.1-8B")
    ax.tick_params(axis='both', which='major')

    for group in desired_order:
        # Check if the group is present in the filtered data
        if group not in filtered_df['color_group'].unique():
            continue  # skip if the group doesn't exist

        group_data = filtered_df[filtered_df["color_group"] == group]

        ax.scatter(
            group_data["bits_per_weight"],
            group_data["ppl"],
            color=color_mapping.get(group, 'black'),
            marker=marker_mapping.get(group, 'o'),
            label=name_mapping.get(group, "unknown"),
            edgecolors='black',
            s=65,
            alpha=0.9
        )



    # Disable scientific notation on the x-axis

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(title="Quantization Type", loc='upper right')
    plt.tight_layout()
    #plt.show()
    plt.savefig("overview_8B.pdf",transparent=True,dpi=300)
    plt.close()


def plot_zfp_8b_chunk(df):

    my_df = df.copy()
    conditions = [
        my_df['quant_type'] == 'F16',
        my_df['quant_type'].isin(['rate', 'accu', 'prec'])
    ]
    choices = ['native', my_df['quant_type']]
    my_df['color_group'] = np.select(conditions, choices, default='built-in')

    my_df["size"] = my_df["size"] / 1024
    filtered_df = my_df[(my_df["ppl"] < 8)
                        & (my_df["quant_type"] != "BF16")
                        & (my_df["color_group"] != "built-in")
                        & (my_df["num_parameter"] == "8B")
                        & (my_df["imat"] == False)
                        & (my_df["size"] < 10)
    ]
    print(my_df)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=250)
    ax.set_xlabel(label_dict["bpw"])
    ax.set_ylabel(label_dict["ppl"])
    ax.set_title("Perplexity of Llama-3.1-8B")
    ax.tick_params(axis='both', which='major')

    desired_order = [1,2,3,4]

    for dim in desired_order:
        # Check if the group is present in the filtered data
        if dim not in filtered_df['dim'].unique():
            continue  # skip if the group doesn't exist

        group_data = filtered_df[filtered_df["dim"] == dim]

        ax.scatter(
            group_data["bits_per_weight"],
            group_data["ppl"],
            color=color_mapping.get(dim, 'black'),
            marker=marker_mapping.get(dim, 'o'),
            label=name_mapping.get(dim, f"{4**dim}"),
            edgecolors='black',
            s=65,
            alpha=0.9
        )



    # Disable scientific notation on the x-axis

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(title="Block size", loc='upper right')
    plt.tight_layout()
    #plt.show()
    plt.savefig("overview_8B_zfp_chunksize.pdf",transparent=True,dpi=300)
    plt.close()


def plt_8b_hellaswag_imatrix(df):

    my_df = df.copy()
    conditions = [
        my_df['quant_type'] == 'F16',
        my_df['quant_type'].isin(['rate', 'accu', 'prec'])
    ]
    choices = ['native', my_df['quant_type']]
    my_df['color_group'] = np.select(conditions, choices, default='built-in')

    my_df["size"] = my_df["size"] / 1024
    filtered_df = my_df[(my_df["hellaswag"] > 55)
                        #& (my_df["ppl"] < 12)
                        & (my_df["quant_type"] != "BF16")
                        & (
                                (my_df["color_group"] == "built-in")
                                | (my_df["color_group"] == "rate")
                        )
                        & (my_df["num_parameter"] == "8B")
                        #& (my_df["imat"] == False)
    ]
    print(my_df)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=250)
    ax.set_xlabel(label_dict["bpw"])
    ax.set_ylabel(label_dict["hellaswag"])
    ax.set_title("HellaSwag Score of Llama-3.1-8B")
    ax.tick_params(axis='both', which='major')
    counter = 1
    for group in desired_order:
        for imat in [True,False]:

            # Check if the group is present in the filtered data
            if group not in filtered_df['color_group'].unique():
                continue  # skip if the group doesn't exist

            group_data = filtered_df[
                (filtered_df["color_group"] == group)
                & (filtered_df["imat"] == imat)
            ]

            ax.scatter(
                group_data["bits_per_weight"],
                group_data["hellaswag"],
                color=color_mapping[counter],
                marker=marker_mapping.get(group, 'o'),
                label=f"{name_mapping.get(group,'unknown').replace(' Quantization','')} + {'with' if imat else 'without'} Importance",
                edgecolors='black',
                s=65,
                alpha=0.9
            )
            counter += 1


    # Disable scientific notation on the x-axis

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))

    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(title="Quantization type", loc='lower right')
    plt.tight_layout()
    #plt.show()
    plt.savefig("overview_8B_hellaswag.pdf",transparent=True,dpi=300)
    plt.close()
if __name__ == "__main__":
    df = pd.read_csv("all_data.csv")
    plot_summary_ppl(df)
    plot_zfp_8b(df)
    plot_zfp_8b_chunk(df)
    plt_8b_hellaswag_imatrix(df)


