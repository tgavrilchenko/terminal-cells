import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
import pandas as pd
import pickle
import os

def dist_to_plane(plane, pts):
    a, b, c = plane

    normalization = np.sqrt(a**2 + b**2 + 1)

    #sq_residuals = [np.abs((a*pt[0] + b*pt[1] + pt[2] + c)/normalization) for pt in pts]
    #print(sq_residuals)
    #tot = np.sum(sq_residuals)
    #print(tot)

    tot = np.sum(np.abs(np.dot(pts, np.transpose([a, b, 1]))/normalization +
                        c*np.ones(len(pts))/normalization))

    return tot

def total_edge_lengths(G):

    tot_length = 0

    for e in G.edges():
        c1 = np.array(G.nodes[e[0]]['coords'][:3])
        c2 = np.array(G.nodes[e[1]]['coords'][:3])
        tot_length += np.sqrt(sum((c1 - c2) ** 2))

    return tot_length

def project_and_rotate(pts):

    N_pts = len(pts)

    center = [np.mean(pts[:, 0]), np.mean(pts[:, 1]), np.mean(pts[:, 2])]
    u, s, vh = np.linalg.svd(pts - np.tile(center, (N_pts, 1)))

    #print(s[0]/s[2], s[1]/s[2])

    v = vh.T[:, 2]
    v = np.array([v[0], v[1], abs(v[2])])
    norm = v / np.sqrt(np.sum(v**2))

    # dist = np.dot(pts, np.transpose(norm)) + np.ones(len(pts))
    # m = np.tile(norm, (N_pts, 1))
    # projected_pts = pts - (m.T * dist).T

    vector = pts - np.tile(center, (N_pts, 1))
    dist = np.dot(vector, np.transpose(norm))

    m = np.tile(norm, (N_pts, 1))
    projected_pts = pts - (m.T * dist).T

    # rotate plane onto the normal [0, 0, 1]

    v = np.zeros((3, 3))
    v[2, 0] = norm[0]
    v[2, 1] = norm[1]
    v[0, 2] = -norm[0]
    v[1, 2] = -norm[1]

    R = np.identity(3) + v + np.dot(v, v) / (1 + norm[2])

    rotated_pts = np.dot(projected_pts, R.T)

    # print('rotated normal', np.dot(norm, R.T))

    #zs = [i[2] for i in rotated_pts]
    # print('mean zs:', np.mean(zs), 'std zs:', np.std(zs))

    return rotated_pts


def project_to_plane_old(fname, get_radii = True):

    n_lines = np.loadtxt(fname)
    N_pts = len(n_lines)

    coords = {int(n_lines[i][0]): n_lines[i][2:5] for i in range(N_pts)}

    # flip y-axis for proper AP orientation
    max_y = np.max([v[1] for v in coords.values()])
    coords = {k: (coords[k][0], max_y - coords[k][1], coords[k][2]) for k in coords.keys()}

    G_3d = nx.Graph()
    G_3d.add_nodes_from([(node, {'coords': coord}) for (node, coord) in coords.items()])

    if get_radii:
        radius = {n_lines[i][0]: n_lines[i][5] for i in range(N_pts)}
        nx.set_node_attributes(G_3d, radius, 'radius')

    for line in n_lines:
        if line[0] in coords and line[1] in coords:
            G_3d.add_edge(int(line[0]), int(line[1]))

    rotated_pts = project_and_rotate(np.array(list(coords.values())))

    new_coords = {i + 1: rotated_pts[i] for i in range(N_pts)}

    G_2d = G_3d.copy()
    nx.set_node_attributes(G_2d, new_coords, 'coords')

    return G_3d, G_2d, total_edge_lengths(G_3d), total_edge_lengths(G_2d) 


    # xs = rotated_pts[:,0]
    # ys = rotated_pts[:,1]
    # area_approx = (np.max(xs) - np.min(xs))*(np.max(ys) - np.min(ys))
    # print(np.max(xs), np.min(xs), np.max(ys), np.min(ys))
    # print('approx:', area_approx)
    # 
    # fig = plt.figure()
    # #ax = fig.add_subplot()
    # ax = fig.add_subplot(projection='3d')
    # 
    # 
    # # for p in range(N_pts):
    # #     if p%10 == 0:
    # #         #ax.scatter(pts[p][0], pts[p][1], pts[p][2], color='b')
    # #         #ax.scatter(new_pts[p][0], new_pts[p][1], new_pts[p][2], color='k')
    # #         ax.scatter(rotated_pts[p][0], rotated_pts[p][1], rotated_pts[p][2], color='k')
    # 
    # plt.plot(new_pts[hull.vertices, 0], new_pts[hull.vertices, 1], 0, 'r--', lw=2)
    # 
    # # fig = plt.figure()
    # #
    # #
    # for p in two_d_pts:
    #     ax.scatter(p[0], p[1], color='k')
    # 
    # # for p in pts:
    # #     ax.scatter(p[0], p[1], p[2], color='b')
    # #
    # #plt.show()

def project_to_plane_from_3d_tree(G_3d, get_radii = True, remove_stem = False):

    coords = nx.get_node_attributes(G_3d, 'coords')

    rotated_pts = project_and_rotate(np.array(list(coords.values())))

    new_coords = {i + 1: rotated_pts[i] for i in range(G_3d.number_of_nodes())}

    G_2d = G_3d.copy()
    nx.set_node_attributes(G_2d, new_coords, 'coords')
    
    if remove_stem:
        this_node = 1
        while G_2d.degree(this_node) != 2:

            #print('removing node', this_node, 'degree:', G_2d.degree(this_node))

            neib = list(G_2d.neighbors(this_node))[0]
            G_2d.remove_node(this_node)
            G_3d.remove_node(this_node)
            this_node = neib

    return G_2d



## BATCH PROJECTION OF TCs
## TEST: save plot comparing 3D and 2D total edge lengths

def batch_projection(TC_file_3D):

    fig, ax = plt.subplots(figsize=(5 , 4))

    colors = [[39/255, 93/255, 173/255], #denim 275dad
              [91/255, 97/255, 106/255], # 5b616a Black coral
              [238/255, 123/255, 48/255], #ee7b30 Princeton orange
              [58/255, 175/255, 185/255]] #3aafb8 verdigris

    instars = [1, 2, 3, 4]
    #instars = [2]

    info_dict = {}
    info_dict['number'] = []
    info_dict['instar'] = []
    info_dict['side'] = []
    info_dict['Tr'] = []
    info_dict['tree'] = []

    phenotype_dir = 'Edge_Lists/mCherry/'

    for instar in instars:
        dir = phenotype_dir + 'L' + str(instar) + '/'
        #dir = 'Edge_Lists/BG2/L' + str(instar) + '/'
        for side in ['R', 'L']:
            for inst in range(50):
                for tr_num in [9]:

                    tree_name = str(inst) + '_Tr' + str(tr_num) + side
                    fname = dir + tree_name + '.txt'

                    if os.path.isfile(fname):
                        print('now on', fname)
                        #save_tree(fname)
                        G_3d, G, len_3d, len_2d = project_to_plane(fname)

                        err =  100*(1 - len_2d/len_3d)

                        plt.scatter(len_3d, err, color = colors[instar-1])

                        info_dict['number'].append(inst)
                        info_dict['instar'].append(instar)
                        info_dict['side'].append(side)
                        info_dict['Tr'].append(tr_num)
                        info_dict['tree'].append(G)


                        # if err < 0.88:
                        #     print("problem with instar", instar, tree_name)


    # plt.xlabel("3D total edge lengths")
    # plt.ylabel("2D total edge lengths")
    # plt.ylabel("percent error in length measurement")
    # plt.savefig("projection_test_edge_comparison_errors.png")
    #
    #
    # dataframe = pd.DataFrame(info_dict, columns = ['number', 'instar', 'side', 'Tr', 'tree'])
    # pickle.dump(dataframe, open(phenotype_dir + '2D_TCs.p', "wb"))