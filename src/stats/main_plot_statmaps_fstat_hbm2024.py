from nilearn.plotting import plot_stat_map, show
import nilearn.image as nl
from matplotlib import cm
stat_img = 'pval_fdr_ftest_TBMsmooth3mm.nii.gz'
pstat_img = '/home/ajoshi/Desktop/ImagePTE/src/stats/pval_fdr_ftest_lesion0mm.nii.gz'
stat_img = '/home/ajoshi/Desktop/ImagePTE/src/stats/fval_lesion0mm.nii.gz'

#stat_img = '/ImagePTE1/ajoshi/code_farm/bfp/src/stats/results/pval_fdr_bord_PTE_smooth0.5_2000.nii.gz'
#stat_img = '/home/ajoshi/projects/bfp/src/stats/results/pval2_fdr_bord_PTE_smooth0.5_sig_temp.nii.gz'
#stat_img = 'pval_KS_lesion1mm.nii.gz'
outfile1 = stat_img.replace('.nii.gz','_1.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile2 = stat_img.replace('.nii.gz','_2.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile3 = stat_img.replace('.nii.gz','_3.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile4 = stat_img.replace('.nii.gz','_4.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile5 = stat_img.replace('.nii.gz','_5.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile6 = stat_img.replace('.nii.gz','_6.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile7 = stat_img.replace('.nii.gz','_7.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile8 = stat_img.replace('.nii.gz','_8.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'


img = 0.05 - nl.load_img(stat_img).get_fdata()
img[img < 0] = 0

msk = nl.load_img(pstat_img).get_fdata() < 0.05
img = nl.load_img(stat_img).get_fdata() * msk

#img = nl.load_img(stat_img).get_fdata()
#img[img < 1e-3] = 0

stat_img_th = nl.new_img_like(stat_img, img)

bk_img = '/home/ajoshi/Software/BrainSuite23a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.nii.gz'

plot_stat_map(stat_img_th,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile1,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

plot_stat_map(stat_img_th,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(35*0.8, 144*0.546875, 210*0.546875),
              display_mode="ortho",
              output_file=outfile2,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

plot_stat_map(stat_img_th,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(30*0.8, 100*0.546875, 180*0.546875),
              display_mode="ortho",
              output_file=outfile3,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

plot_stat_map(stat_img_th,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(25*0.8, 66*0.546875, 150*0.546875),
              display_mode="ortho",
              output_file=outfile4,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')


# Plot stat maps without thresholding

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile5,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(35*0.8, 144*0.546875, 210*0.546875),
              display_mode="ortho",
              output_file=outfile6,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(30*0.8, 100*0.546875, 180*0.546875),
              display_mode="ortho",
              output_file=outfile7,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

plot_stat_map(stat_img,
              bk_img,
              threshold=0.0,
              draw_cross=False,
              cut_coords=(25*0.8, 66*0.546875, 150*0.546875),
              display_mode="ortho",
              output_file=outfile8,
              annotate=True,dim=-0.5,vmax=5,vmin=0,cmap='hot')

