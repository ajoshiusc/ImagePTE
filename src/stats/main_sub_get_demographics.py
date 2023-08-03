import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
import csv


f = np.load('PTE_graphs_USCLobes.npz')
sub_ids_pte = f['sub_ids']

f = np.load('NONPTE_graphs_USCLobes.npz')
sub_ids_nonpte = f['sub_ids']

sub_ids=np.concatenate((sub_ids_nonpte,sub_ids_pte))

csvfname='/ImagePTE1/ajoshi/code_farm/ImagePTE/src/stats/ImagePTE_Maryland.csv'
# Import BrainSync libraries

sub_ID = []
sub_fname = []
subAtlas_idx = []
reg_var = []
reg_cvar1 = []
reg_cvar2 = []
count1 = 0


with open(csvfname, newline='') as csvfile:
    dialect = csv.Sniffer().sniff(next(open(csvfname)))
    creader = csv.DictReader(csvfile, delimiter=dialect.delimiter, quotechar='"')
    
    for row in creader:
        sub = row['subID']
        rvar = row['PTE']
        rcvar1 = row['Age']
        rcvar2 = row['Gender']

        if rcvar2 == 'M' or rcvar2 == 'Male':
            rcvar2 = 0
        elif rcvar2 == 'F' or rcvar2 == 'Female':
            rcvar2 = 1
        else:
            rcvar2 = 0.5

        sub_ID.append(sub)
        reg_var.append(float(rvar))
        reg_cvar1.append(float(rcvar1))
        reg_cvar2.append(float(rcvar2))
        count1 += 1

print('CSV file read\nThere are %d subjects' % (len(sub_ID)))

num_pte=0
num_nonpte=0

num_pte_females = 0
num_nonpte_females = 0

for sub in sub_ids:
    print(sub)

    ind = sub_ID.index(sub)

    if reg_var[ind]>0.5:
        num_pte += 1
        num_pte_females += reg_cvar2[ind]

    else:
        num_nonpte += 1
        num_nonpte_females += reg_cvar2[ind]







print(num_pte, num_nonpte, num_pte_females, num_nonpte_females)
