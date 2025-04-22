import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use("default")
plt.rcParams.update({'figure.facecolor': 'white','axes.facecolor': 'white'})
plt.rc('font', family='serif')
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"]

# Labels and corresponding colors
labels = [
    "light-ops",
    "heavy-ops",
    "machine-clear",
    "branch-mispredict",
    "memory-bound",
    "core-bound",
    "fetch-bandwidth",
    "backend-latency",
]

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

# 80% opacity on each color bar
handles = [plt.Line2D([0], [0], color=color, lw=10, alpha=0.60) for color in colors]

# Optional: Nimbus Roman font

# Create your legend-only figure
fig_legend = plt.figure(figsize=(5, 3))

# Set a border on the figure patch (the background)
fig_legend.patch.set_edgecolor('black')  # Set the border color
fig_legend.patch.set_linewidth(2)          # Set the border width

# Create the legend as before
fig_legend.legend(
    handles,
    labels,
    loc="center",
    frameon=False,  # This is for the legend box; the figure border is set separately.
    ncol=1,
    handlelength=2.5,
    title="Topdown L2 Metrics",
)

fig_legend.tight_layout()
fig_legend.show()
fig_legend.savefig("topdown_l2_legend.svg",
                   format="svg",
                   bbox_inches='tight',
                   transparent=False)

