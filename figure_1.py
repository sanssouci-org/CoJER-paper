#%%
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
alpha = 0.1
data_ = np.load('results/coverages_openml_fcp_control_multiple_splits.npy')
data = np.sum(data_ > 1 - alpha, axis=1) / data_.shape[1]
# Define variables
nb_datasets, nb_methods, nb_models = data.shape

# Model and method names
models = ["RF", "Lasso", "MLP", "SVR", "KNN"]
methods = ["CoJER", "Gazin et al.", "SCP"]

# Assign a color for each method
colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

# Prepare data for plotting
# We need to rearrange the data so that each group of boxplots can be plotted together
boxplot_data = [data[:, method, model] for model in range(nb_models) for method in range(nb_methods)]

# Create the boxplot
fig, ax = plt.subplots(figsize=(4, 2.5))


# Positions for the groups and the width of each group
gap = 0.8
positions = []
for i in range(nb_models):
    positions.extend([i * (nb_methods + gap) + j for j in range(nb_methods)])
# Boxplot
ax.boxplot(boxplot_data, sym='', positions=positions, widths=0.6)
ax.hlines(1 - alpha, -0.1, nb_models * (nb_methods + 0.5), colors='r', linestyles='dashed', label=f'Level {1 - alpha}')

# Add scatter plot for each data point
for model_idx in range(nb_models):
    for method_idx in range(nb_methods):
        # Get data points
        y = data[:, method_idx, model_idx]
        # Calculate the x position for scatter
        pos = model_idx * (nb_methods + gap) + method_idx
        x = np.random.normal(pos, 0.1, size=len(y))
        # Scatter plot with method-specific colors
        ax.scatter(x, y, alpha=0.75, marker='v', color=colors[method_idx], label=f'{methods[method_idx]}' if model_idx == 0 else "")

# Customizing the plot
ax.set_xticks([(nb_methods / 2) + (nb_methods + gap) * i for i in range(nb_models)])
ax.set_xticklabels(models)
ax.set_ylabel('FCP event control')
ax.set_title('FCP control')

# Adding a legend for the methods
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='w', markerfacecolor=colors[i], marker='v', markersize=10, label=methods[i]) for i in range(nb_methods)]
# ax.legend(handles=legend_elements, loc='upper right')
plt.savefig('results/fig_FCP_control_v2.pdf', bbox_inches='tight')
plt.show()

# %%

data_ = np.load('results/lengths_openml_fcp_control_mutiple_splits.npy')
best_int = np.min(data_, axis=2)
div = np.repeat(best_int[:, :, np.newaxis, :], repeats=nb_methods, axis=2)

data = data_ / div
data = np.mean(data, axis=1)

plt.rc('legend', fontsize=10)
# Define variables
nb_datasets, nb_methods, nb_models = data.shape

# Model and method names
models = ["RF", "Lasso", "MLP", "SVR", "KNN"]
methods = ["CoJER", "Gazin et al.", "SCP"]

# Assign a color for each method
colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

# Prepare data for plotting
# We need to rearrange the data so that each group of boxplots can be plotted together
boxplot_data = [data[:, method, model] for model in range(nb_models) for method in range(nb_methods)]

# Create the boxplot
fig, ax = plt.subplots(figsize=(4, 2.5))

# Positions for the groups and the width of each group
gap = 0.8
positions = []
for i in range(nb_models):
    positions.extend([i * (nb_methods + gap) + j for j in range(nb_methods)])
# Boxplot
ax.boxplot(boxplot_data, sym='', positions=positions, widths=0.6)
ax.set_ylim(0.95, 2)

# Add scatter plot for each data point
for model_idx in range(nb_models):
    for method_idx in range(nb_methods):
        # Get data points
        y = data[:, method_idx, model_idx]
        # Calculate the x position for scatter
        pos = model_idx * (nb_methods + gap) + method_idx
        x = np.random.normal(pos, 0.1, size=len(y))
        # Scatter plot with method-specific colors
        ax.scatter(x, y, alpha=0.75, marker='v', color=colors[method_idx], label=f'{methods[method_idx]}' if model_idx == 0 else "")

# Customizing the plot
ax.set_xticks([(nb_methods / 2) + (nb_methods + gap) * i for i in range(nb_models)])
ax.set_xticklabels(models)
ax.set_title("Interval lengths")
ax.set_ylabel('Relative length (vs best)')

from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(lambda y, _: f'{y:.1f}')
ax.yaxis.set_major_formatter(formatter)

# Adding a legend for the methods
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='w', markerfacecolor=colors[i], marker='v', markersize=10, label=methods[i]) for i in range(nb_methods)]
ax.legend(handles=legend_elements, loc='upper right')
plt.savefig('results/fig_FCP_control_v2_lengths.pdf', bbox_inches='tight')
plt.show()
# %%
