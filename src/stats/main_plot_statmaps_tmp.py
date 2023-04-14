from nilearn.plotting import plot_stat_map, show
import nilearn.image as nl
from matplotlib import cm
stat_img = 'pval_fdr_ftest_TBMsmooth3mm.nii.gz'
pstat_img = 'pval_fdr_ftest_lesion0mm.nii.gz'
stat_img = 'fval_lesion0mm.nii.gz'

#stat_img = '/ImagePTE1/ajoshi/code_farm/bfp/src/stats/results/pval_fdr_bord_PTE_smooth0.5_2000.nii.gz'
#stat_img = '/home/ajoshi/projects/bfp/src/stats/results/pval2_fdr_bord_PTE_smooth0.5_sig_temp.nii.gz'
#stat_img = 'pval_KS_lesion1mm.nii.gz'
outfile1 = stat_img.replace('.nii.gz','_1.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile2 = stat_img.replace('.nii.gz','_2.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile3 = stat_img.replace('.nii.gz','_3.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile4 = stat_img.replace('.nii.gz','_4.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'

msk = nl.load_img(pstat_img).get_fdata() < 0.05
img = nl.load_img(stat_img).get_fdata() * msk
#img = nl.load_img(stat_img).get_fdata()
#img[img < 1e-3] = 0

stat_img = nl.new_img_like(stat_img, img)

bk_img = '/home/ajoshi/BrainSuite21a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.nii.gz'

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile1,
              annotate=True,dim=0,vmax=5)

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile2,
              annotate=True,dim=-0.5,vmax=5)

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile3,
              annotate=True,dim=-0.75,vmax=5)

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile4,
              annotate=True,dim=-1,vmax=5)

