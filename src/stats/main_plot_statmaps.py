from nilearn.plotting import plot_stat_map, show
import nilearn.image as nl
from matplotlib import cm
stat_img = 'pval_fdr_ftest_TBMsmooth3mm.nii.gz'
stat_img = 'pval_fdr_ftest_lesion.smooth3mm.nii.gz'
stat_img = '/ImagePTE1/ajoshi/code_farm/bfp/pval_fdr_bord_PTE_smooth1.5_150.nii.gz'
outfile1 = stat_img.replace('.nii.gz','_1_new.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile2 = stat_img.replace('.nii.gz','_2_new.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile3 = stat_img.replace('.nii.gz','_3_new.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile4 = stat_img.replace('.nii.gz','_4_new.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'

img = 0.05 - nl.load_img(stat_img).get_data()

img[img < 0] = 0

stat_img = nl.new_img_like(stat_img, img)

bk_img = '/home/ajoshi/BrainSuite19b/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.nii.gz'

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(40*0.8, 188*0.546875, 240*0.546875),
              display_mode="ortho",
              output_file=outfile1,
              annotate=True)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(35*0.8, 144*0.546875, 210*0.546875),
              display_mode="ortho",
              output_file=outfile2,
              annotate=True)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(30*0.8, 100*0.546875, 180*0.546875),
              display_mode="ortho",
              output_file=outfile3,
              annotate=True)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(25*0.8, 66*0.546875, 150*0.546875),
              display_mode="ortho",
              output_file=outfile4,
              annotate=True)

