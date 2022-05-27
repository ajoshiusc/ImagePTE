from tqdm import tqdm
import scipy.io as spio
from brainsync import normalizeData
import scipy as sp


def load_bfp_data(sub_fname, LenTime):
    ''' sub_fname: list of filenames of .mat files that contains Time x Vertex matrix of subjects' preprocessed fMRI data '''
    ''' LenTime: number of timepoints in data. this should be the same in all subjects '''
    ''' Outputs 3D matrix: Time x Vector x Subjects '''
    count1 = 0
    subN = len(sub_fname)
    print('loading data for ' + str(subN) + ' subjects')
    pbar = tqdm(total=subN)
    for ind in range(subN):
        fname = sub_fname[ind]
        df = spio.loadmat(fname)
        data = df['dtseries'].T
        if int(data.shape[0]) != LenTime:
            print(sub_fname[ind] +
                  ' has %d timepoints, while %d were expected' %
                  (data.shape[0], LenTime))
        d, _, _ = normalizeData(data[:LenTime, ])

        if count1 == 0:
            sub_data = sp.zeros((LenTime, d.shape[1], subN))

        sub_data[:, :, count1] = d
        count1 += 1
        pbar.update(1)
        if count1 == subN:
            break

    pbar.close()

    print('loaded data for ' + str(subN) + ' subjects')
    return sub_data
