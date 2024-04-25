from tree_projection import project_to_plane_from_3d_tree
import networkx as nx
from scipy.spatial import ConvexHull
import numpy as np
from get_features import compute_void, edge_lengths, total_edge_length, coarsen_graph

import sys
import pickle
import pandas as pd

import matplotlib.pyplot as plt

info_dict = {}
info_dict['number'] = []
info_dict['instar'] = []
info_dict['side'] = []
info_dict['Tr'] = []
info_dict['tree'] = []
info_dict['tree_3d'] = []
info_dict['tree_coarsened'] = []

info_dict['eLen_3D'] = []
info_dict['eLen_2D'] = []
info_dict['eLen_2D_tips'] = []
info_dict['hull_area'] = []
info_dict['norm_shape_param'] = []
info_dict['n_nodes'] = []
info_dict['n_leaves'] = []
info_dict['n_edges'] = []
info_dict['mean_void_size'] = []
info_dict['std_void_size'] = []
info_dict['trunk_len'] = []
info_dict['avg_e_len'] = []
info_dict['max_path_len'] = []


# dir = 'TC_data/'
# fname = 'retraced_mCherry_TCs_raw'
# #fname = 'mCherry_TCs_raw'

dir = 'TC_data/hybrid_data/'
fname = 'hybrid_mCherry_TCs_raw'

with open(dir + fname + '.p', 'rb') as f:
    data = pickle.load(f)

#data = data[(data['instar'] <= 2)]

for index, row in data.iterrows():

    number = row['number']
    instar = int(row['instar'])
    side = row['side']
    Tr = row['Tr']
    G_3d = row['tree']

    tree_name = str(instar) + '_Tr' + str(Tr) + side

    print(tree_name)

    G_2d = project_to_plane_from_3d_tree(G_3d, remove_stem = False)

    for e in G_2d.edges():
        c1 = np.array(G_2d.nodes[e[0]]['coords'])
        c2 = np.array(G_2d.nodes[e[1]]['coords'])
        G_2d[e[0]][e[1]]['length'] = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

        c1 = np.array(G_3d.nodes[e[0]]['coords'])
        c2 = np.array(G_3d.nodes[e[1]]['coords'])
        G_3d[e[0]][e[1]]['length'] = np.sqrt((c1[0] - c2[0]) ** 2 +
                                             (c1[1] - c2[1]) ** 2 +
                                             (c1[2] - c2[2]) ** 2)


    elen_3d = np.sum(list(nx.get_edge_attributes(G_3d, 'length').values()))
    elen_2d = np.sum(list(nx.get_edge_attributes(G_2d, 'length').values()))

    #print('new root is:', min(G_2d.nodes()))
    #print('nodes:', G_2d.number_of_nodes())

    coarsened_G = coarsen_graph(G_2d, root = min(G_2d.nodes()))

    print('nodes here:', G_2d.number_of_nodes())

    n2 = coarsened_G.number_of_nodes()
    e1 = coarsened_G.number_of_edges()

    neibs = [len(coarsened_G[n].keys()) for n in coarsened_G.nodes()]
    leaves2 = len(np.where(np.array(neibs) == 1)[0])

    # print('should be equal:', leaves1, leaves2)

    coords = nx.get_node_attributes(G_2d, 'coords')

    pts = [(i[0], i[1]) for i in coords.values()]
    hull = ConvexHull(pts)
    area = hull.volume

    perimeter = hull.area
    max_s_param = 0.5 / np.sqrt(np.pi)
    norm_shape_param = np.sqrt(area) / perimeter / max_s_param
    #print('shape param:', np.sqrt(area) / perimeter / max_s_param)

    #trunk_len, avg_e_len, len_tips, max_path_len = edge_lengths(H, G)

    voids = compute_void(G_2d)
    void_avg = np.mean(voids)
    void_std = np.std(voids)

    if instar == 1:
        c = 5
    elif instar == 2:
        c = 10
    else:
        c = 15

    #G_2d_coarsened = coarsen_graph(G_2d, max_separation = c, root = min(G.nodes()))

    # print(G_2d.number_of_nodes(), 'nodes')
    #
    # for e in G_2d.edges():
    #     c0 = G_2d.nodes[e[0]]['coords']
    #     c1 = G_2d.nodes[e[1]]['coords']
    #     plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color = 'k', linewidth = 2)
    #
    # for n in G_2d.nodes():
    #     plt.text(G_2d.nodes[n]['coords'][0], G_2d.nodes[n]['coords'][1], n, color = 'r', size = 5)
    #
    # plt.show()
    # sys.exit()

    info_dict['number'].append(number)
    info_dict['instar'].append(instar)
    info_dict['side'].append(side)
    info_dict['Tr'].append(Tr)
    info_dict['tree'].append(G_2d)
    info_dict['tree_3d'].append(G_3d)
    info_dict['tree_coarsened'].append(coarsened_G)

    info_dict['n_nodes'].append(n2)
    info_dict['n_leaves'].append(leaves2)
    info_dict['n_edges'].append(e1)

    info_dict['eLen_3D'].append(elen_3d)
    info_dict['eLen_2D'].append(elen_2d)
    #info_dict['eLen_2D_tips'].append(np.round(0.001 * len_tips, 6))

    info_dict['hull_area'].append(area)
    info_dict['norm_shape_param'].append(np.round(norm_shape_param, 6))
    info_dict['mean_void_size'].append(void_avg)
    info_dict['std_void_size'].append(void_std)

    # info_dict['trunk_len'].append(trunk_len)
    # info_dict['avg_e_len'].append(avg_e_len)
    # info_dict['max_path_len'].append(max_path_len)

    #plt.scatter(elen_3d, 100*(1 - elen_2d/elen_3d))



dataframe = pd.DataFrame(info_dict, columns=['number', 'instar', 'side', 'Tr', 'tree', 'tree_3d', \
                                             'tree_coarsened', 'eLen_3D', 'eLen_2D', \
                                             'hull_area', \
                                             'norm_shape_param', \
                                             'n_nodes', 'n_leaves', 'n_edges', \
                                             'mean_void_size', 'std_void_size'])
#pickle.dump(dataframe, open(dir + 'retraced_mCherry_TCs_projected_analyzed.p', "wb"))
pickle.dump(dataframe, open(dir + 'hybrid_mCherry_TCs_projected_analyzed_with_stems.p', "wb"))


# plt.xlabel("3D total edge lengths")
# plt.ylabel("2D total edge lengths")
# plt.ylabel("percent error in length measurement")
# plt.savefig("projection_test_edge_comparison_errors_new_traces_removed_stems.png")