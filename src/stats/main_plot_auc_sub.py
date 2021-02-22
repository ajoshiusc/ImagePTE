
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.load('auc_num_sub3.npz')

auc_mean_sub = a['auc_mean_sub']
auc_std_sub = a['auc_std_sub']
auc_sub_all = a['auc_sub_all']
df = pd.DataFrame(auc_sub_all.T).melt(var_name='Num Sub', value_name='AUC')

a_plot = sns.lineplot(data=df, ci='sd', x='Num Sub', y='AUC', linewidth=3)
a_plot.set(ylim=(0, 1))
a_plot.set(xlim=(23, 50))

plt.show()
