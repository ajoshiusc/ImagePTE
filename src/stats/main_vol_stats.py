import pandas as pd
import glob
import os
import shutil
from multiprocessing import Pool
from itertools import product, repeat
import numpy as np
import numbers
from shutil import copyfile, copy
import time

def main():

    tbi_done_list = '/big_disk/ajoshi/fitbir/preproc/maryland_rao_v1_epilepsy_imgs.txt'

    with open(tbi_done_list) as f:
        epiIds = f.readlines()

    epiIds = list(map(lambda x: x.strip(), epiIds))
    print(epiIds.index)


if __name__ == "__main__":
    main()
