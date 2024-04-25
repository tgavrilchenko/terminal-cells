from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def project(fname, savename):

    im = io.imread(fname)
    A = np.array(im)
    res = A.max(axis=0)

    fig, ax = plt.subplots()

    plt.imshow(res, cmap=plt.cm.gray_r)

    plt.axis('equal')
    plt.axis('off')
    plt.savefig(savename, bbox_inches = 'tight')
    plt.close()


def individual_projections():

    for instar in [1, 2, 3]:

        top = 'L1_L2_L3_dataset/'
        dir = top + 'masks_L' + str(instar) + '/'
        savedir = top + 'all_individual_fills/'

        all_files = os.listdir(dir)
        all_files = sorted(all_files)

        for f in all_files:

            fname = f.split('.')[0]

            print(fname)

            if len(fname) > 2:
                project(dir + f, savedir + 'L' + str(instar) + '_' + fname + '.pdf')

def double_projections():

    for instar in [1, 2, 3]:

        top = 'L1_L2_L3_dataset/'
        dir = top + 'masks_L' + str(instar) + '/'
        savedir = top + 'fills_L' + str(instar) + '/'

        idents = np.arange(1, 30, 1)

        Trs = [9]

        all_files = os.listdir(dir)

        for i in idents:

            for Tr in Trs:

                left = str(i) + '_Tr' + str(Tr) + 'L.tif'
                right = str(i) + '_Tr' + str(Tr) + 'R.tif'

                if left in all_files and right in all_files:

                    print(str(i) + '_Tr' + str(Tr))

                    A = np.array(io.imread(dir + left))
                    L = A.max(axis=0)

                    L = L[~np.all(L == 0, axis=1)]
                    L = L[:, ~np.all(L == 0, axis=0)]

                    Ly = len(L)
                    Lx = len(L[0])

                    A = np.array(io.imread(dir + right))
                    R = A.max(axis=0)

                    R = R[~np.all(R == 0, axis=1)]
                    R = R[:, ~np.all(R == 0, axis=0)]

                    Ry = len(R)
                    Rx = len(R[0])

                    # print(len(L), len(L[0]))
                    # print(len(R), len(R[0]))


                    b = 20

                    if Ly < Ry:
                        newL = np.zeros((Ry, Lx))
                        newL[:Ly, :Lx] = L

                        buff = np.zeros((Ry, b))

                        canvas = np.concatenate((newL, buff, R), axis = 1)

                    else:
                        newR = np.zeros((Ly, Rx))
                        newR[:Ry, :Rx] = R

                        buff = np.zeros((Ly, b))

                        canvas = np.concatenate((L, buff, newR), axis = 1)

                    fig, ax = plt.subplots(figsize=(3, 3))

                    # L3
                    if instar == 3:
                        plt.xlim((-100, 1100))
                        plt.ylim((1100, -100))

                    # L2
                    elif instar == 2:
                        plt.xlim((-100, 700))
                        plt.ylim((700, -100))

                    # L1
                    else:
                        plt.xlim((-100, 500))
                        plt.ylim((500, -100))

                    plt.imshow(canvas, cmap=plt.cm.gray_r)

                    savename = savedir + str(i) + '_Tr' + str(Tr) + '.pdf'

                    #plt.axis('equal')
                    plt.axis('off')
                    plt.savefig(savename, bbox_inches='tight')
                    plt.close()


double_projections()



