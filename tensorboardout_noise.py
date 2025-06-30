from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
import numpy as np
import os
import pandas as pd
from numpy import pi

######
#
# Single-aperture
#
######

# Set the main path
directory_path_sa = 'final_results/singleaperture_aug/'

###
#
# Plot the error histograms
#
###
# Plot the error histogram for test A
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# Augmentation data on augmentation trained model
# Test
file_name = 'sa_aug_hist_alpha.csv'
full_path = os.path.join(directory_path_sa, file_name)
df_hist_a_good = pd.read_csv(full_path)

df_hist_a_good['Model'] = 'Augmented model'

# Augmentation data on non-augmentation trained model
# Test
file_name = 'sa_aug_noaugmodel_hist_alpha.csv'
full_path = os.path.join(directory_path_sa, file_name)
df_hist_a_bad = pd.read_csv(full_path)

df_hist_a_bad['Model'] = 'Non-augmented model'

# Combine all dataframes
df_histcat_a = pd.concat([df_hist_a_bad, df_hist_a_good])

# Plot
sns.set_style("ticks")
palette = [sns.color_palette("hls")[0], sns.color_palette("hls")[2]]
sns.histplot(data=df_histcat_a, x='Error', bins=1000, stat='probability', kde=True,
             hue='Model', palette=palette, multiple='layer')
plt.xlim(0, 0.06)
#plt.ylim(0, 0.1)
plt.xlabel('Absolute error $\\alpha$/$^{\circ}$')
plt.ylabel('Probability')
sns.despine()
# save the plot as PDF file
plt.savefig("sa_hista_noise.pdf", format='pdf', bbox_inches = "tight")
plt.show()

# # Plot the error histogram for test B
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# Augmentation data on augmentation trained model
# Test
file_name = 'sa_aug_hist_beta.csv'
full_path = os.path.join(directory_path_sa, file_name)
df_hist_b_good = pd.read_csv(full_path)

df_hist_b_good['Model'] = 'Augmented model'

# Augmentation data on non-augmentation trained model
# Test
file_name = 'sa_aug_noaugmodel_hist_beta.csv'
full_path = os.path.join(directory_path_sa, file_name)
df_hist_b_bad = pd.read_csv(full_path)

df_hist_b_bad['Model'] = 'Non-augmented model'

# Combine all dataframes
df_histcat_b = pd.concat([df_hist_b_bad, df_hist_b_good])

# Plot
sns.set_style("ticks")
palette = [sns.color_palette("hls")[0], sns.color_palette("hls")[2]]
sns.histplot(data=df_histcat_b, x='Error', bins=1000, stat='probability', kde=True,
             hue='Model', palette=palette, multiple='layer')
plt.xlim(0, 0.06)
#plt.ylim(0, 0.1)
plt.xlabel('Absolute error $\\beta$/$^{\circ}$')
plt.ylabel('Probability')
sns.despine()
# save the plot as PDF file
plt.savefig("sa_histb_noise.pdf", format='pdf', bbox_inches = "tight")
plt.show()

######
#
# Multi-aperture
#
######

# Set the main path
directory_path_ma = 'final_results/multiaperture_aug/'

###
#
# Plot the error histograms
#
###
# Plot the error histogram for test A
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# Augmentation data on augmentation trained model
# Test
file_name = 'ma_aug_hist_alpha.csv'
full_path = os.path.join(directory_path_ma, file_name)
df_hist_a_good_ma = pd.read_csv(full_path)

df_hist_a_good_ma['Model'] = 'Augmented model'

# Augmentation data on non-augmentation trained model
# Test
file_name = 'ma_aug_noaugmodel_hist_alpha.csv'
full_path = os.path.join(directory_path_ma, file_name)
df_hist_a_bad_ma = pd.read_csv(full_path)

df_hist_a_bad_ma['Model'] = 'Non-augmented model'

# Combine all dataframes
df_histcat_a_ma = pd.concat([df_hist_a_bad_ma, df_hist_a_good_ma])

# Plot
sns.set_style("ticks")
palette = [sns.color_palette("hls")[0], sns.color_palette("hls")[2]]
sns.histplot(data=df_histcat_a_ma, x='Error', bins=100, stat='probability', kde=True,
             hue='Model', palette=palette, multiple='layer', linewidth=0.04)
plt.xlim(0, 0.1)
#plt.ylim(0, 0.1)
plt.xlabel('Absolute error $\\alpha$/$^{\circ}$')
plt.ylabel('Probability')
sns.despine()
# save the plot as PDF file
plt.savefig("ma_hista_noise.pdf", format='pdf', bbox_inches = "tight")
plt.show()

# # Plot the error histogram for test B
sns.set_theme()
sns.set_context("paper")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 1.5})
# Augmentation data on augmentation trained model
# Test
file_name = 'ma_aug_hist_beta.csv'
full_path = os.path.join(directory_path_ma, file_name)
df_hist_b_good_ma = pd.read_csv(full_path)

df_hist_b_good_ma['Model'] = 'Augmented model'

# Augmentation data on non-augmentation trained model
# Test
file_name = 'ma_aug_noaugmodel_hist_beta.csv'
full_path = os.path.join(directory_path_ma, file_name)
df_hist_b_bad_ma = pd.read_csv(full_path)

df_hist_b_bad_ma['Model'] = 'Non-augmented model'

# Combine all dataframes
df_histcat_b_ma = pd.concat([df_hist_b_bad_ma, df_hist_b_good_ma])

# Plot
sns.set_style("ticks")
palette = [sns.color_palette("hls")[0], sns.color_palette("hls")[2]]
sns.histplot(data=df_histcat_b_ma, x='Error', bins=100, stat='probability', kde=True,
             hue='Model', palette=palette, multiple='layer', linewidth=0.04)
plt.xlim(0, 0.1)
#plt.ylim(0, 0.1)
plt.xlabel('Absolute error $\\beta$/$^{\circ}$')
plt.ylabel('Probability')
sns.despine()
# save the plot as PDF file
plt.savefig("ma_histb_noise.pdf", format='pdf', bbox_inches = "tight")
plt.show()