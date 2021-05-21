import numpy as np
import pandas as pd
import pdb


def load_csv(path="/big_disk/ajoshi/ADHD_Peking_gord/",
             csv_file="/big_disk/ajoshi/ADHD_Peking_bfp/Peking_all_phenotypic.csv"):
    df = pd.read_csv(csv_file)
    sub_ids = np.array(df['ScanDir ID'])
    labels = np.array(df['DX'])
    print(labels)
    pdb.set_trace()
    labels[labels > 0] = 1

    sub_names = []
    for id in sub_ids:
       sub_names.append(path + str(id) + "_rest_bold.32k.GOrd.mat")

    return sub_names