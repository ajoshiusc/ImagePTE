from matplotlib import cm
from nilearn.plotting import plot_stat_map
import nilearn.image as nl
#stat_img = 'pval_fdr_ftest_TBMsmooth3mm.nii.gz'
#stat_img = 'pval_fdr_ftest_lesion.smooth3mm.nii.gz'
#stat_img = '/ImagePTE1/ajoshi/code_farm/bfp/src/stats/results/pval_fdr_bord_PTE_smooth0_2000.nii.gz'
pstat_img = '/home/ajoshi/Projects/ImagePTE/src/alff_analysis/pval2_alff_bord_PTE_smooth0.5_sig_perm.nii.gz'
stat_img_fname = '/home/ajoshi/Projects/ImagePTE/src/alff_analysis/fval2_bord_PTE_smooth0.5_sig_perm.nii.gz'
#stat_img = 'pval_KS_lesion1mm.nii.gz'
# '/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile1 = stat_img_fname.replace('.nii.gz', '_1.png')
# '/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile2 = stat_img_fname.replace('.nii.gz', '_2.png')
# '/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile3 = stat_img_fname.replace('.nii.gz', '_3.png')
# '/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'
outfile4 = stat_img_fname.replace('.nii.gz', '_4.png')

img = nl.load_img(stat_img_fname).get_fdata()

pimg = 0.1 - nl.load_img(pstat_img).get_fdata()
img[pimg <= 0] = 0

#img = nl.load_img(stat_img).get_fdata()
#img[img < 1e-3] = 0

stat_img = nl.new_img_like(stat_img_fname, img)


stat_img.to_filename(stat_img_fname.replace('.nii.gz', '_thresh.nii.gz'))

stat_img = stat_img_fname.replace('.nii.gz', '_thresh.nii.gz')

bk_img = '/home/ajoshi/Software/BrainSuite21a/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.nii.gz'

plot_stat_map(stat_img,
              bk_img,
              threshold=0.,
              draw_cross=False,
              cut_coords=(124, 104, 128),
              display_mode="ortho",
              output_file=outfile1,symmetric_cbar=False,cmap='hot',
              annotate=True, dim=-0.5, vmax=5)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(107, 57, 137),
              display_mode="ortho",
              output_file=outfile2,symmetric_cbar=False,cmap='hot',
              annotate=True, dim=-0.5, vmax=5)

plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(81, 59, 113),
              display_mode="ortho",
              output_file=outfile3,symmetric_cbar=False,cmap='hot',
              annotate=True, dim=-0.5, vmax=5)
