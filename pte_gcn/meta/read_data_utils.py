import numpy as np


def load_features(root_path):

    f = np.load(root_path + 'PTE_fmridiff_USCLobes.npz')
    conn_pte = f['fdiff_sub']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']
    n_rois = conn_pte.shape[0]
    epi_brainsync = conn_pte.T

    f = np.load(root_path + 'PTE_graphs_USCLobes.npz')
    conn_pte = f['conn_mat']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']
    n_rois = conn_pte.shape[0]
    ind = np.tril_indices(n_rois, k=1)
    epi_connectivity = conn_pte[ind[0], ind[1], :].T

    a = np.load(root_path + 'PTE_lesion_vols_USCLobes.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    epi_lesion_vols = np.array([a[k] for k in sub_ids])
    epi_measures = np.concatenate(
        (.3*epi_lesion_vols, epi_connectivity, .3*epi_brainsync), axis=1)

    f = np.load(root_path + 'NONPTE_fmridiff_USCLobes.npz')
    conn_pte = f['fdiff_sub']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']
    n_rois = conn_pte.shape[0]
    nonepi_brainsync = conn_pte.T

    f = np.load(root_path + 'NONPTE_graphs_USCLobes.npz')
    conn_nonpte = f['conn_mat']
    lab_ids = f['label_ids']
    gordlab = f['labels']
    sub_ids = f['sub_ids']

    nonepi_connectivity = conn_nonpte[ind[0], ind[1], :].T

    a = np.load(root_path + 'NONPTE_lesion_vols_USCLobes.npz', allow_pickle=True)
    a = a['lesion_vols'].item()
    nonepi_lesion_vols = np.array([a[k] for k in sub_ids])
    nonepi_measures = np.concatenate(
        (.3*nonepi_lesion_vols, nonepi_connectivity, .3*nonepi_brainsync), axis=1)


    X = np.vstack((epi_measures, nonepi_measures))
    y = np.hstack(
        (np.ones(epi_measures.shape[0]), np.zeros(nonepi_measures.shape[0])))

    return X, y