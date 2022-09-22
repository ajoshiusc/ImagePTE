import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import pdb
import nibabel as nib
import numpy as np

# import os
# import SimpleITK as sitk

def show_slices(img):
    # img = np.transpose(img, [2,1,0])
    w, h, l = img.shape

    slices = [img[w // 2, :, :], img[:, h // 2, :], img[:, :, l // 2]]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice, cmap='gray', origin='lower')

def plot_heatmap(pte_data, non_data):
    pte_conn = pte_data['TE_mat']
    non_conn = non_data['TE_mat']
    print(pte_conn.shape)
    for i in range(pte_conn.shape[0]):
        th = np.percentile(np.abs(non_conn[i]).reshape(-1), 98)
        temp = non_conn[i]
        temp[temp < th] = 0
        print(th, np.where(temp > 0)[0].shape, pte_conn)
        plt.imshow(temp, cmap='hot')
        plt.colorbar()
        pdb.set_trace()
        plt.show()


# def check_commun(graph):
#     for i in range(graph.shape[0]):
#         if graph[]

def find_common_graph(conn, num_sub):
    conn[conn < 0.4] = 0
    conn[conn > 0] = 1

    common_graph = np.zeros_like(conn[0])

    for i in range(conn.shape[1]):
        for j in range(conn.shape[2]):
            if np.sum(conn[:, i, j]) > num_sub:
                common_graph[i, j] = 1
                # common_graph[j, i] = 1

    np.set_printoptions(threshold=np.inf)
    print(np.sum(common_graph))

    # print(conn[3])
    # plt.imshow(common_graph)
    # plt.show()
    # pdb.set_trace()
    return common_graph


def get_graphs_from_mask(common_graph, conn):
    coords = np.where(common_graph == 0)
    for i in range(conn.shape[0]):
        conn[i][coords] = 0
    # print(conn[0])
        plt.imshow(conn[i])
        plt.show()
    # pdb.set_trace()
    return conn

file_pos = "/home/wenhuicu/data_npz/PTE_Allconn_BCI-DNI.npz"
file_neg = "/home/wenhuicu/data_npz/NONPTE_Allconn_BCI-DNI.npz"
#
pte_data = np.load(file_pos, encoding='bytes', allow_pickle=True)
non_data = np.load(file_neg, encoding='bytes', allow_pickle=True)

conn_pte = np.abs(pte_data["conn_mat"])

mask_pte = find_common_graph(conn_pte[5:25], 10)
#
# conn_pte_sparse = get_graphs_from_mask(common_graph_pte, np.abs(pte_data["conn_mat"]))


conn_non = np.abs(non_data["conn_mat"])

mask_non = find_common_graph(conn_non[10:30], 10)
temp = mask_non == mask_pte
diff = np.zeros_like(mask_non)
diff[np.where(temp==False)] = 1
print(np.sum(diff))
plt.imshow(diff)
plt.show()
# get_graphs_from_mask(mask, np.abs(non_data["conn_mat"]))
# file_lesion_pte = "/ImagePTE1/ajoshi/code_farm/ImagePTE/src/stats/PTE_lesion_vols.npz"
# file_lesion_non = "/ImagePTE1/ajoshi/code_farm/ImagePTE/src/stats/NONPTE_lesion_vols.npz"
#
# lesion_pte = np.load(file_lesion_pte, encoding='bytes', allow_pickle=True)
# lesion_non = np.load(file_lesion_non, encoding='bytes', allow_pickle=True)
atlas = nib.load('/ImagePTE1/ajoshi/fitbir/preproc/maryland_rao_v1/TBI_INVAE281PAQ/vae_mse.flair.atlas.mask.nii.gz').get_data()
# a = lesion_pte['lesion_vols']
pdb.set_trace()
# show_slices(atlas)
# plt.show()
# plot_heatmap(pte_data, non_data)