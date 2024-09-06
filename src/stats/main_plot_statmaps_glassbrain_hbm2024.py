import cmd
from nilearn.plotting import plot_stat_map, show
import nilearn.image as nl
from matplotlib import cm
from nilearn import plotting

stat_img = 'pval_fdr_ftest_TBMsmooth3mm.nii.gz'
pstat_img = '/home/ajoshi/Desktop/ImagePTE/src/stats/pval_fdr_ftest_lesion0mm.nii.gz'
stat_img = '/home/ajoshi/Desktop/ImagePTE/src/stats/fval_lesion0mm.nii.gz'

stat_img_mni = '/home/ajoshi/Desktop/fval_lesion0mm_mni.nii.gz'

apply_map_exe = '/home/ajoshi/Software/BrainSuite23a/svreg/bin/svreg_apply_map.sh'

#bci2mni_map = '/home/ajoshi/Software/BrainSuite23a/svreg/BCI-DNI_brain_atlas/mni2bci.map.nii.gz'
bci2mni_map = '/deneb_disk/icbm152/mni_icbm152_nlin_asym_09c/icbm152_t1.svreg.map.nii.gz'

target_file = '/deneb_disk/icbm152/mni_icbm152_nlin_asym_09c/icbm152_t1.GM_frac.nii.gz' #'/home/ajoshi/Software/BrainSuite23a/svreg/BrainSuiteAtlas1/mri.bfc.nii.gz'
#target_file = '/home/ajoshi/Software/fsl/data/atlases/MNI/MNI-maxprob-thr50-1mm.nii.gz'

cmd = apply_map_exe + ' ' + bci2mni_map + ' ' + stat_img + ' ' + stat_img_mni + ' ' + target_file

print(cmd)

import os
os.system(cmd)


plotting.plot_glass_brain(stat_img_mni, threshold=3)
plotting.show()



plotting.plot_glass_brain(
    stat_img_mni,
    threshold=0,
    vmax=5,
    display_mode="lyrz",
    cmap="hot",
    colorbar=True,
    plot_abs=False,
    title="Lesion Glass Brain",
    output_file="/home/ajoshi/Desktop/lesion_glassbrain_hbm2024.png",
)
plotting.show()
