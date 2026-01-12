import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sc_path = r"data\processed\sc_connectome\schaefer400\sub-NDARINV0A4ZDYNL_ses-2YearFollowUpYArm1.csv"

# 读取为纯数值数组
data = pd.read_csv(sc_path, header=None, index_col=None).values

# 取前400×400并取log(1+x)
sc_400 = np.log1p(data[:400, :400])
sns.heatmap(sc_400)
plt.xlabel("Region")
plt.ylabel("Connectivity")
plt.title("Schaefer 400 Connectome")
plt.show()
