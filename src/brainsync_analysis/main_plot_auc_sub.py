
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.load('auc_num_sub_brainsync.npz')

auc_mean_sub = a['auc_mean_sub']
auc_std_sub = a['auc_std_sub']
auc_sub_all = a['auc_sub_all']
#auc_sub_all = auc_sub_all[25:,:]

df = pd.DataFrame(auc_sub_all.T).melt(var_name='Num Sub', value_name='AUC')

a_plot = sns.lineplot(data=df, ci='sd', x='Num Sub', y='AUC', linewidth=7)
#sns.regplot(x="Num Sub", y="AUC", data=df,order=2,ci=None,scatter=False)
#sns.regplot(x="Num Sub", y="AUC", data=df,order=1,ci=None,scatter=False)

#a_plot.set_xscale('log')
a_plot.set(ylim=(0.0, 1))

xmin=25
xmax=50+1

a_plot.set(xlim=(2, xmax))
x = df.values[100*xmin:, 0]#range(xmin,xmax+1)
y = df.values[100*xmin:, 1]

ylim=np.ones(1)
xlim=np.arange(100,100+1)
x=np.concatenate((x,xlim))
y=np.concatenate((y,ylim))


p = np.polyfit(x=x, y=y, deg=2)
y = np.polyval(p, x=range(xmin,xmax))
plt.plot(range(xmin,xmax),y)
#df = pd.DataFrame(auc_sub_all.T).melt(var_name='Num Sub', value_name='AUC')

plt.show()
