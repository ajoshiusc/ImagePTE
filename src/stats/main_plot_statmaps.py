from nilearn.plotting import plot_stat_map, show
import nilearn.image as nl

stat_img = '/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.nii.gz'
outfile = stat_img.replace('.nii.gz','.png') #'/home/ajoshi/coding_ground/ImagePTE/src/stats/pval_hotelling.smooth3mm.png'

img = 0.05 - nl.load_img(stat_img).get_data()

img[img < 0] = 0

stat_img = nl.new_img_like(stat_img, img)

bk_img = '/home/ajoshi/coding_ground/svreg/BCI-DNI_brain_atlas/BCI-DNI_brain.nii.gz'
plot_stat_map(stat_img,
              bk_img,
              threshold=0,
              draw_cross=False,
              cut_coords=(100, 100, 100),
              display_mode="ortho",
              output_file=outfile,
              annotate=False)

