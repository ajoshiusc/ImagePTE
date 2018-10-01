#||AUM||
#||Shree Ganeshaya Namaha||

import pandas as pd


def main():

    print('hi')

    med_hist_csv = '/big_disk/ajoshi/fitbir/tracktbi_pilot/Baseline Med History_246/TrackTBI_MedicalHx.csv'

    print(med_hist_csv)
    subIds = pd.read_csv(med_hist_csv) #, index_col=1)
    print(subIds)
    ''' If fMRI data exists for some subjects, then store their cognitive scores '''
    for subid in self.subids:
        cog_scores.append(self.get_cog_score_subid(subid))


if __name__ == "__main__":
    main()
