import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
plt.style.use("default")
plt.rcParams.update({'figure.facecolor': 'white','axes.facecolor': 'white'})
plt.rc('font', family='serif')
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"]

df = pd.read_csv("runtimes.csv")
patterns = ['//', '\\\\', '///', '\\\\\\','/','\\']
# # --- 1a. Create a new grouping column.
# # For rows with quant_type "rate", concatenate quant_type, dim, threshold_low, threshold_high.
# def create_group(row):
#     if row["quant_type"].strip().lower() == "rate":
#         return f"{row['quant_type']}_{row['dim']}_{row['threshold_low']}_{row['threshold_high']}"
#     else:
#         return row["quant_type"]
#
# df["group"] = df.apply(create_group, axis=1)
#
# # --- 2. Compute aggregated statistics based on the new 'group' column.
#
# # For throughput: Group by the new group column.
# throughput_stats = df.groupby("group").agg(
#     prompt_median=("prompt_eval_throughput", "median"),
#     prompt_min=("prompt_eval_throughput", "min"),
#     prompt_max=("prompt_eval_throughput", "max"),
#     eval_median=("eval_throughput", "median"),
#     eval_min=("eval_throughput", "min"),
#     eval_max=("eval_throughput", "max")
# ).reset_index()
#
# # Compute error bars: lower error = median - min, upper error = max - median.
# throughput_stats["prompt_err_low"] = throughput_stats["prompt_median"] - throughput_stats["prompt_min"]
# throughput_stats["prompt_err_high"] = throughput_stats["prompt_max"] - throughput_stats["prompt_median"]
# throughput_stats["eval_err_low"]   = throughput_stats["eval_median"] - throughput_stats["eval_min"]
# throughput_stats["eval_err_high"]  = throughput_stats["eval_max"] - throughput_stats["eval_median"]
#
# # For evaluation runtime grouped by the new group and ncore.
# runtime_stats = df.groupby(["group", "ncore"]).agg(
#     eval_time_median=("eval_time", "median"),
#     eval_time_min=("eval_time", "min"),
#     eval_time_max=("eval_time", "max")
# ).reset_index()
#
# runtime_stats["time_err_low"] = runtime_stats["eval_time_median"] - runtime_stats["eval_time_min"]
# runtime_stats["time_err_high"] = runtime_stats["eval_time_max"] - runtime_stats["eval_time_median"]
#
# # --- 3. Plot Throughput (prompt_eval and eval) per group as a grouped bar plot
#
# fig, ax = plt.subplots(figsize=(12, 6))
# x = np.arange(len(throughput_stats))
# width = 0.35
#
# bars1 = ax.bar(x - width/2, throughput_stats["prompt_median"], width,
#                yerr=[throughput_stats["prompt_err_low"], throughput_stats["prompt_err_high"]],
#                capsize=5, label="Prompt Throughput")
#
# bars2 = ax.bar(x + width/2, throughput_stats["eval_median"], width,
#                yerr=[throughput_stats["eval_err_low"], throughput_stats["eval_err_high"]],
#                capsize=5, label="Eval Throughput")
# ax.set_yscale('log')
# ax.set_xticks(x)
# ax.set_xticklabels(throughput_stats["group"], rotation=45, ha="right")
# ax.set_ylabel("Throughput (tokens/sec)")
# ax.set_title("Throughput (Prompt and Eval) by Group")
# ax.legend()
# plt.tight_layout()
# plt.show()
#
# # --- 4. Plot Eval Runtime over 'ncore' per group as a grouped bar plot
#
# # Get unique ncore values sorted.
# ncore_values = sorted(runtime_stats["ncore"].unique())
# groups = runtime_stats["group"].unique()
#
# fig, ax = plt.subplots(figsize=(12, 6))
# width = 0.15  # Bar width; adjust according to the number of groups
#
# # Create offsets for each group to avoid bars overlapping.
# offsets = {grp: (i - (len(groups)-1)/2) * width for i, grp in enumerate(groups)}
#
# # Plot bars for each group.
# for grp in groups:
#     sub = runtime_stats[runtime_stats["group"] == grp]
#     # Compute positions based on ncore values.
#     pos = np.array([ncore_values.index(n) for n in sub["ncore"]]) + offsets[grp]
#     ax.bar(pos, sub["eval_time_median"], width,
#            yerr=[sub["time_err_low"], sub["time_err_high"]],
#            capsize=5, label=f"{grp}")
#
#
# ax.set_xticks(np.arange(len(ncore_values)))
# ax.set_xticklabels(ncore_values)
# ax.set_xlabel("ncore")
# ax.set_ylabel("Evaluation Runtime (ms)")
# ax.set_title("Evaluation Runtime by ncore for each Group")
# ax.legend(ncol=2, fontsize="small")
# plt.tight_layout()
# plt.show()


# -------------------------------
# 1a. Create a new "group" column.
def create_group(row):
    if row["quant_type"].strip().lower() == "rate":
        return f"{row['quant_type']}_{row['dim']}_{row['threshold_low']}_{row['threshold_high']}"
    else:
        return row["quant_type"]

df["group"] = df.apply(create_group, axis=1)

# (Optional) if you wish to rename groups for better presentation, you can define a dictionary.
rename_dict = {
    "rate": "ZFP-Rate",
    "rate_2.0_4.0_4.0": "Rate:4-Block:16",
    "rate_2.0_8.0_8.0": "Rate:8-Block:16",
    "rate_3.0_4.0_4.0": "Rate:4-Block:64",
    "rate_3.0_8.0_8.0": "Rate:8-Block:64",
    "rate_4.0_4.0_4.0": "Rate:4-Block:256",
    "rate_4.0_8.0_8.0": "Rate:8-Block:256",
}
# You can apply renaming after aggregation if needed.

# -------------------------------
# 2. Compute aggregated throughput statistics by group.
throughput_stats = df[df["ncore"]==96]
throughput_stats = throughput_stats.groupby(["group","threshold_low"],dropna=False).agg(
    prompt_median=("prompt_eval_throughput", "median"),
    prompt_min=("prompt_eval_throughput", "min"),
    prompt_max=("prompt_eval_throughput", "max"),
    eval_median=("eval_throughput", "median"),
    eval_min=("eval_throughput", "min"),
    eval_max=("eval_throughput", "max")
).reset_index()



# Compute error bars: lower error = median - min, upper error = max - median.
throughput_stats["prompt_err_low"] = throughput_stats["prompt_median"] - throughput_stats["prompt_min"]
throughput_stats["prompt_err_high"] = throughput_stats["prompt_max"] - throughput_stats["prompt_median"]
throughput_stats["eval_err_low"]   = throughput_stats["eval_median"] - throughput_stats["eval_min"]
throughput_stats["eval_err_high"]  = throughput_stats["eval_max"] - throughput_stats["eval_median"]

# (Optional) Rename the group names using our rename_dict.
throughput_stats["group"] = throughput_stats["group"].replace(rename_dict)
throughput_stats = throughput_stats[
    (
        ( throughput_stats["group"] == "Q4_0" )
        | ( throughput_stats["group"] == "Q4_0" )
        #| ( throughput_stats["group"] == "Q4_1" )
        #| ( throughput_stats["group"] == "Q4_K_M" )
        | ( throughput_stats["group"] == "Q6_K" )
        | ( throughput_stats["group"] == "Q8_0" )
        | (
                (throughput_stats["threshold_low"] == 4.0 )
                |(throughput_stats["threshold_low"] == 8.0 )
          )
    )
]
# -------------------------------
# 3. Compute aggregated runtime statistics by group and ncore.
runtime_stats = df.groupby(["group", "ncore","quant_type","dim","threshold_low"],dropna=False).agg(
    eval_time_median=("eval_time", "median"),
    eval_time_min=("eval_time", "min"),
    eval_time_max=("eval_time", "max")
).reset_index()

runtime_stats["time_err_low"] = runtime_stats["eval_time_median"] - runtime_stats["eval_time_min"]
runtime_stats["time_err_high"] = runtime_stats["eval_time_max"] - runtime_stats["eval_time_median"]

# If desired, rename the "group" values in runtime_stats as well.
runtime_stats["group"] = runtime_stats["group"].replace(rename_dict)

# -------------------------------
# 4. Plot: Throughput (prompt and eval) as a grouped bar plot.
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=250)#fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(throughput_stats))
width = 0.35

ax.bar(x - width/2, throughput_stats["prompt_median"], width,
       #yerr=[throughput_stats["prompt_err_low"], throughput_stats["prompt_err_high"]],
       capsize=5, label="Prefill Throughput", alpha=0.7, hatch="///")
ax.bar(x + width/2, throughput_stats["eval_median"], width,
       #yerr=[throughput_stats["eval_err_low"], throughput_stats["eval_err_high"]],
       capsize=5, label="Decode Throughput", alpha=0.7, hatch="\\\\\\")


ax.set_xticks(x)
ax.set_xticklabels(throughput_stats["group"], rotation=30, ha="right")
ax.set_xlabel("Quantization Type")
ax.set_ylabel("Throughput [token/s]")
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
ax.set_yticks([0.1,1,10,100])

ax.set_title("Throughput Llama-3.1-8B (96 Threads)")
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
#plt.show()
plt.savefig("throughput-8B.pdf", transparent=True)
plt.close()
# -------------------------------
# 5. Plot: Evaluation Runtime vs ncore for each group (grouped bar plot).
# Get sorted unique ncore values.
ncore_values = sorted(runtime_stats["ncore"].unique())
runtime_stats_filtered = runtime_stats[
    (
        (runtime_stats["group"] =="Q4_0")
        | (runtime_stats["group"] =="Q6_K")
        | (runtime_stats["group"] =="Q8_0")

        | (
            ((runtime_stats["threshold_low"] == 4.0 )
            |(runtime_stats["threshold_low"] == 8.0 ))
            & (runtime_stats["dim"] ==3)
      )
    )
    #&
   # (
        #(runtime_stats["dim"] ==3)
        #| (runtime_stats["dim"] =="")
    #)
]
groups = runtime_stats_filtered["group"].unique()

#fig, ax = plt.subplots(figsize=(12, 6))
fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=250)#fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.15
# Compute offsets for each group.
offsets = {grp: (i - (len(groups)-1)/2) * bar_width for i, grp in enumerate(groups)}

for pattern, grp in zip(patterns,groups):

    sub = runtime_stats_filtered[runtime_stats_filtered["group"] == grp]
    pos = np.array([ncore_values.index(n) for n in sub["ncore"]]) + offsets[grp]
    ax.bar(pos, sub["eval_time_median"]/1000, bar_width,
           #yerr=[sub["time_err_low"], sub["time_err_high"]],
           capsize=5, label=grp, alpha=0.7,hatch=pattern)

ref_scale_x = np.arange(1, 4.05, 1/24, ) - 1 + 0.3 #np.arange(24, 96, 1) / 10
ref_scale_y = 3800/1000 / np.arange(1, 4.05, 1/24)
ax.plot(ref_scale_x, ref_scale_y, linestyle="-", color="k", label="Reference: linear scaling")


ax.set_yscale('log')
ax.set_xticks(np.arange(len(ncore_values)))
ax.set_xticklabels(ncore_values)
ax.set_xlabel("Number of Threads")
ax.set_yticks([x/1000 for x in [100, 200, 1000, 3000, 12000]])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
ax.set_ylabel("Time per token [s]")
ax.set_title("Runtime per decode token for Llama-3.1-8B")
ax.legend(ncol=1, fontsize="small", loc='upper right')
ax.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
#plt.show()
plt.savefig("runtime_decode-8B.pdf", transparent=True)
