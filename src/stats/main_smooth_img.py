from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import nilearn.image as ni

atlas = '/home/ajoshi/BrainSuite19a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.bfc.nii.gz'
a = np.load('Hotelling_results.npz')
pval = a['pval']
msk = a['msk']

ati = ni.load_img(atlas)

pval_vol = np.zeros(ati.shape)

h, pval_fdr = fdrcorrection(pval, alpha=0.15, method='indep')
pval_vol = pval_vol.flatten()
pval_vol[msk] = pval<0.05
pval_vol = pval_vol.reshape(ati.shape)

# Save pval
pvalnii = ni.new_img_like(ati, pval_vol)
pvalnii.to_filename('pval_hotelling.nii.gz')

pvalnii = ni.smooth_img(pvalnii, 5)
pvalnii.to_filename('pval_hotelling_smooth5.nii.gz')


pvalnii = ni.smooth_img(pvalnii, 10)
pvalnii.to_filename('pval_hotelling_smooth10.nii.gz')


print(pval.min())

print('done')
