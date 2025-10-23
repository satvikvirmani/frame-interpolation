import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file = pd.read_csv("../video_metrics/metrics_output.csv")

corr_matrix = csv_file[["MSE", "Inverse SSIM", "LPIPS"]].corr()

print("Correlation matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
plt.title("Correlation Matrix")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.show()