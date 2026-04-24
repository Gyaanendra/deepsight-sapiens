import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to your results
csv_path = r"E:\deepsight\runs\detect\KnightSight_Local\RTX4060_Uniform_v1\results.csv"
output_dir = "docs/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns] # Clean column names

# Set visual style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.facecolor'] = '#0e1117'
plt.rcParams['axes.facecolor'] = '#0e1117'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Create a multi-plot figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('🚀 YOLO Detection Training Performance', fontsize=20, color='#92FE9D')

# 1. Box & Object Loss
sns.lineplot(data=df, x='epoch', y='train/box_loss', ax=axes[0,0], label='Train Box Loss', color='#00C9FF')
sns.lineplot(data=df, x='epoch', y='val/box_loss', ax=axes[0,0], label='Val Box Loss', color='#92FE9D')
axes[0,0].set_title('Localization Loss (Precision)')

# 2. Classification Loss
sns.lineplot(data=df, x='epoch', y='train/cls_loss', ax=axes[0,1], label='Train Cls Loss', color='#00C9FF')
sns.lineplot(data=df, x='epoch', y='val/cls_loss', ax=axes[0,1], label='Val Cls Loss', color='#92FE9D')
axes[0,1].set_title('Classification Loss (Recall)')

# 3. mAP 50
sns.lineplot(data=df, x='epoch', y='metrics/mAP50(B)', ax=axes[1,0], color='#92FE9D', linewidth=3)
axes[1,0].set_title('Mean Average Precision (mAP@50)')
axes[1,0].set_ylim(0, 1.05)

# 4. mAP 50-95
sns.lineplot(data=df, x='epoch', y='metrics/mAP50-95(B)', ax=axes[1,1], color='#FFD700', linewidth=3)
axes[1,1].set_title('Precision @ Higher IoU (mAP@50-95)')
axes[1,1].set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = os.path.join(output_dir, "training_metrics.png")
plt.savefig(plot_path, facecolor='#0e1117')
print(f"Visualization saved to: {plot_path}")
