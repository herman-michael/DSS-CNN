from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
import numpy as np
import os
import pandas as pd
from numpy import pi

# Set the main path
directory_path = 'final_results/multiaperture_noaug/'
directory_path_aug = 'final_results/multiaperture_aug/'

###
#
# Plot the learning rate
#
###
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# No augmentation
# Training
file_name = 'lr.csv'
full_path = os.path.join(directory_path, file_name)
df_lr = pd.read_csv(full_path)
df_lr['Case - Dataset'] = 'No aug - Train'
# Augmentation
# Training
file_name = 'lr.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_lr_aug = pd.read_csv(full_path)
df_lr_aug['Case - Dataset'] = 'Aug - Train'

# Combine all dataframes
df_catlr = pd.concat([df_lr_aug, df_lr])

# Get the last value of 'Step' in the augmented training data
max_step_aug_train = df_lr_aug['Step'].max()

sns.set_style("white")
palette = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3]]
sns.lineplot(data=df_catlr, x='Step', y='Value',
             hue='Case - Dataset', palette=palette)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.xlim(0, max_step_aug_train)
plt.ylabel('Learning rate')
plt.xlabel('Steps')
# save the plot as PDF file
plt.savefig("ma_lr.pdf", format='pdf', bbox_inches = "tight")
plt.show()

###
#
# Plot the loss
#
###
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# No augmentation
# Training
file_name = 'loss_train.csv'
full_path = os.path.join(directory_path, file_name)
df_train = pd.read_csv(full_path)
df_train['Case - Dataset'] = 'No aug - Train'

# Validation
file_name = 'loss_valid.csv'
full_path = os.path.join(directory_path, file_name)
df_valid = pd.read_csv(full_path)
df_valid['Case - Dataset'] = 'No aug - Valid'

# Augmentation
# Training
file_name = 'loss_train.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_train_aug = pd.read_csv(full_path)
df_train_aug['Case - Dataset'] = 'Aug - Train'

# Validation
file_name = 'loss_valid.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_valid_aug = pd.read_csv(full_path)
df_valid_aug['Case - Dataset'] = 'Aug - Valid'

# Combine all dataframes
df = pd.concat([df_train_aug, df_valid_aug, df_train, df_valid])

# Get the last value of 'Step' in the augmented training data
max_step_aug_train = df_train_aug['Step'].max()

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(8, 4))
loss = sns.lineplot(data=df, x='Step', y='Value', hue='Case - Dataset', palette=sns.color_palette("Paired", 4))

# Set alpha for 'Aug - Train' and 'No aug - Train' lines to 0.6
for line, label in zip(loss.lines, df['Case - Dataset'].unique()):
    if label in ['Aug - Train', 'No aug - Train']:
        plt.setp(line, alpha=0.7)

plt.grid(True, which="both", ls="-")
plt.xlim(0, max_step_aug_train)
plt.ylim(10e-9, 1e-5)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Steps')
# save the plot as PDF file
plt.savefig("ma_loss.pdf", format='pdf', bbox_inches = "tight")
plt.show()

###
#
# Plot the R2
#
###
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# No augmentation
# Training
file_name = 'r2_train.csv'
full_path = os.path.join(directory_path, file_name)
df_train_r2 = pd.read_csv(full_path)
df_train_r2['Case - Dataset'] = 'No aug - Train'

# Validation
file_name = 'r2_valid.csv'
full_path = os.path.join(directory_path, file_name)
df_valid_r2 = pd.read_csv(full_path)
df_valid_r2['Case - Dataset'] = 'No aug - Valid'

# Augmentation
# Training
file_name = 'r2_train.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_train_r2_aug = pd.read_csv(full_path)
df_train_r2_aug['Case - Dataset'] = 'Aug - Train'

# Validation
file_name = 'r2_valid.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_valid_r2_aug = pd.read_csv(full_path)
df_valid_r2_aug['Case - Dataset'] = 'Aug - Valid'

# Combine all dataframes
df_r2 = pd.concat([df_train_r2_aug, df_valid_r2_aug, df_train_r2, df_valid_r2])

# Get the last value of 'Step' in the augmented training data
max_step_aug_train = df_train_r2_aug['Step'].max()

# Plot
sns.set_style("whitegrid")
loss = sns.lineplot(data=df_r2, x='Step', y='Value', hue='Case - Dataset', palette=sns.color_palette("Paired", 4))

# Set alpha for 'Aug - Train' and 'No aug - Train' lines to 0.6
for line, label in zip(loss.lines, df_r2['Case - Dataset'].unique()):
    if label in ['Aug - Train', 'No aug - Train']:
        plt.setp(line, alpha=0.7)

plt.grid(True, which="both", ls="-")
plt.xlim(0, max_step_aug_train)
plt.ylim(0.999, 1.0001)
#plt.yscale('log')
plt.ylabel('R-squared')
plt.xlabel('Steps')
# save the plot as PDF file
plt.savefig("ma_r2.pdf", format='pdf', bbox_inches = "tight")
plt.show()

###
#
# Plot the heat map
#
###
# Plot the error heatmap A
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# Load the error heatmap as a csv file
file_name = 'ma_noaug_errorimage_alpha.csv'
full_path = os.path.join(directory_path, file_name)
df_heatmap_a = pd.read_csv(full_path)
imagea = df_heatmap_a.values
sns.set_style("ticks")
heatmap_a = sns.heatmap(imagea, cmap='flare', vmin=0, vmax=0.03, linewidths=0.0)
heatmap_a.set_xticks([0, 256, 512])
heatmap_a.set_xticklabels(['$-\\theta/2$', '0', '$\\theta/2$'])
heatmap_a.set_yticks([0, 256, 512])
heatmap_a.set_yticklabels(['$\\theta/2$', '0', '$-\\theta/2$'])
heatmap_a.set_xlabel('$\\alpha$/$^{\circ}$')
heatmap_a.set_ylabel('$\\beta$/$^{\circ}$')
colorbar = heatmap_a.collections[0].colorbar
colorbar.set_label('Absolute error $\\alpha$/$^{\circ}$')
plt.xticks(rotation=0) 
# save the plot as PDF file
plt.savefig("ma_hma.png", format='png', dpi=400, bbox_inches = "tight")
plt.show()

# Plot the error heatmap B
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# Load the error heatmap as a csv file
file_name = 'ma_noaug_errorimage_beta.csv'
full_path = os.path.join(directory_path, file_name)
df_heatmap_b = pd.read_csv(full_path)
imageb = df_heatmap_b.values
sns.set_style("ticks")
heatmap_b = sns.heatmap(imageb, cmap='flare', vmin=0, vmax=0.03, linewidths=0.0)
heatmap_b.set_xticks([0, 256, 512])
heatmap_b.set_xticklabels(['$-\\theta/2$', '0', '$\\theta/2$'])
heatmap_b.set_yticks([0, 256, 512])
heatmap_b.set_yticklabels(['$\\theta/2$', '0', '$-\\theta/2$'])
heatmap_b.set_xlabel('$\\alpha$/$^{\circ}$')
heatmap_b.set_ylabel('$\\beta$/$^{\circ}$')
colorbar = heatmap_b.collections[0].colorbar
colorbar.set_label('Absolute error $\\beta$/$^{\circ}$')
plt.xticks(rotation=0) 
# save the plot as PDF file
plt.savefig("ma_hmb.png", format='png', dpi=400, bbox_inches = "tight")
plt.show()

###
#
# Plot the error histograms
#
###
# Plot the error histogram for test A
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# No augmentation
# Test
file_name = 'ma_noaug_hist_alpha.csv'
full_path = os.path.join(directory_path, file_name)
df_hist_a = pd.read_csv(full_path)
df_hist_a['Case - Dataset'] = 'No aug - Test'

# Augmentation
# Test
file_name = 'ma_aug_hist_alpha.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_hist_a_aug = pd.read_csv(full_path)
df_hist_a_aug['Case - Dataset'] = 'Aug - Test'

# Combine all dataframes
df_histcat_a = pd.concat([df_hist_a_aug, df_hist_a])

# Plot
sns.set_style("ticks")
palette = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3]]
sns.histplot(data=df_histcat_a, x='Error', bins=75, stat='probability', kde=True,
             hue='Case - Dataset', palette=palette, multiple='dodge')
plt.xlim(0, 0.03)
plt.ylim(0, 0.08)
plt.xlabel('Absolute error $\\alpha$/$^{\circ}$')
plt.ylabel('Probability')
sns.despine()
# save the plot as PDF file
plt.savefig("ma_hista.pdf", format='pdf', bbox_inches = "tight")
plt.show()

# Plot the error histogram for test B
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# No augmentation
# Test
file_name = 'ma_noaug_hist_beta.csv'
full_path = os.path.join(directory_path, file_name)
df_hist_b = pd.read_csv(full_path)
df_hist_b['Case - Dataset'] = 'No aug - Test'

# Augmentation
# Test
file_name = 'ma_aug_hist_beta.csv'
full_path = os.path.join(directory_path_aug, file_name)
df_hist_b_aug = pd.read_csv(full_path)
df_hist_b_aug['Case - Dataset'] = 'Aug - Test'

# Combine all dataframes
df_histcat_b = pd.concat([df_hist_b_aug, df_hist_b])

# Plot
sns.set_style("ticks")
palette = [sns.color_palette("Paired")[1], sns.color_palette("Paired")[3]]
sns.histplot(data=df_histcat_b, x='Error', bins=75, stat='probability', kde=True,
             hue='Case - Dataset', palette=palette, multiple='dodge')
plt.xlim(0, 0.03)
plt.ylim(0, 0.08)
plt.xlabel('Absolute error $\\beta$/$^{\circ}$')
plt.ylabel('Probability')
sns.despine()
# save the plot as PDF file
plt.savefig("ma_histb.pdf", format='pdf', bbox_inches = "tight")
plt.show()

###
#
# Plot the predicted vs actual diagrams
#
###
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
min_size = 20
max_size = 200
# Plot the predicted vs actual for test A
file_name = 'ma_noaug_actvspred_alpha.csv'
full_path = os.path.join(directory_path, file_name)
df_actvspred_a = pd.read_csv(full_path)
df_actvspred_a['Target'] = df_actvspred_a['Target'] * 180/pi
df_actvspred_a['Output'] = df_actvspred_a['Output'] * 180/pi
df_actvspred_a['Error'] = df_actvspred_a['Error'] * 180/pi
sizes_a = min_size + (df_actvspred_a['Error'] - df_actvspred_a['Error'].min()) * (max_size - min_size) / (df_actvspred_a['Error'].max() - df_actvspred_a['Error'].min())
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
xtrue = np.linspace(-4.5,4.5,512)
ytrue = xtrue
plt.plot(xtrue, ytrue, linewidth=1, color='white', zorder=2)
pva_a = plt.scatter(data=df_actvspred_a, x='Target', y='Output', s=sizes_a, facecolors='none',
            c='Error', cmap='flare', alpha=0.5, zorder=1)
plt.xlabel('Actual $\\alpha$/$^{\circ}$')
plt.ylabel('Predicted $\\alpha$/$^{\circ}$')
cbar = plt.colorbar(pva_a)
cbar.set_label('Absolute error $\\alpha$/$^{\circ}$')
pva_a.set_clim(0,0.05)
plt.legend(["Ideal" , "Predictions - Test"])
# save the plot as PDF file
plt.savefig("ma_pvaa.pdf", format='pdf', bbox_inches = "tight")
plt.show()

# Plot the predicted vs actual for test B
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
file_name = 'ma_noaug_actvspred_beta.csv'
full_path = os.path.join(directory_path, file_name)
df_actvspred_b = pd.read_csv(full_path)
df_actvspred_b['Target'] = df_actvspred_b['Target'] * 180/pi
df_actvspred_b['Output'] = df_actvspred_b['Output'] * 180/pi
df_actvspred_b['Error'] = df_actvspred_b['Error'] * 180/pi
sizes_b = min_size + (df_actvspred_b['Error'] - df_actvspred_b['Error'].min()) * (max_size - min_size) / (df_actvspred_b['Error'].max() - df_actvspred_b['Error'].min())
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
xtrue = np.linspace(-4.5,4.5,512)
ytrue = xtrue
plt.plot(xtrue, ytrue, linewidth=1, color='white', zorder=2)
pva_b = plt.scatter(data=df_actvspred_b, x='Target', y='Output', s=sizes_b, facecolors='none',
            c='Error', cmap='flare', alpha=0.5, zorder=1)
plt.xlabel('Actual $\\beta$/$^{\circ}$')
plt.ylabel('Predicted $\\beta$/$^{\circ}$')
cbar = plt.colorbar(pva_b)
cbar.set_label('Absolute error $\\beta$/$^{\circ}$')
pva_b.set_clim(0,0.05)
plt.legend(["Ideal" , "Predictions - Test"])
# save the plot as PDF file
plt.savefig("ma_pvab.pdf", format='pdf', bbox_inches = "tight")
plt.show()