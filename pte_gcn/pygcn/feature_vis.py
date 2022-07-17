import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
import pandas as pd

root_path = '/home/wenhuicu/ImagePTE-1/'

def plot_3d(data1, data2):
    feat1 = np.mean(data1, axis=1)
    feat2 = np.mean(data2, axis=1)
  
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feat1[:, 0], feat1[:, 1], feat1[:, 2])
    ax.scatter(feat2[:, 0], feat2[:, 1], feat2[:, 3])
    plt.savefig("features_3d_2")


def plot_pairwise(data1, data2):
    feat1 = np.mean(data1, axis=1)
    feat2 = np.mean(data2, axis=1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feat1[:, 0], feat1[:, 1], feat1[:, 2])
    ax.scatter(feat2[:, 0], feat2[:, 1], feat2[:, 3])
    plt.savefig("features_pair")


def plot_parallel(data1, data2):
    ss = StandardScaler()
    feat1 = np.mean(data1, axis=1)
    feat2 = np.mean(data2, axis=1)
    df = pd.DataFrame(np.concatenate([feat1[:, 0:64:4], feat2[:, 0:64:4]], axis=0))
    # pdb.set_trace()
    # df = ss.fit_transform(df)
    class_col = ['PTE']*36 + ['NONPTE']*36
    df['class'] = class_col
    df.head()

    # plot parallel coordinates
    pc = parallel_coordinates(df, 'class', color=('#FFE888', '#FF9999'))
    plt.savefig('features_parallel_15feats_065', format='eps')


if __name__ == '__main__':
    data1 = np.load(root_path + 'PTE_deepwalk1605_brain2.npz')['deepwalk']
    data2 = np.load(root_path + 'NONPTE_deepwalk1605_brain2.npz')['deepwalk']
    plot_parallel(data1, data2)
