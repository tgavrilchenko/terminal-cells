import numpy as np
import networkx as nx
import sys
import pickle
import pandas as pd
import time

from generate_networks import BSARW
from helpers import total_edge_length, convex_hull_area, compute_void, number_of_branches

def run_ensemble(sizes, s, b, elen = 1, len_init = 30):

    latency_dist = 0

    for max_size in sizes:

        G = BSARW(max_size, elen, stretch_factor=s, branch_probability=b,
                        initial_len=len_init, init='line', right_side_only=True)

        coords = np.array(list(nx.get_node_attributes(G, 'coords').values()))
        # G_maxs = coords.max(axis=0)
        # G_mins = coords.min(axis=0)

        voids = compute_void(G)

        # if save_tree_opt:
        #     info_dict['tree'].append(G)
        info_dict['N'].append(max_size)
        info_dict['s'].append(s)
        info_dict['b'].append(b)
        info_dict['hull_area'].append(convex_hull_area(G))
        info_dict['tot_len'].append(total_edge_length(G))
        info_dict['n_edges'].append(number_of_branches(G))
        info_dict['void_mean'].append(np.mean(voids))
        info_dict['void_std'].append(np.std(voids))

        # info_dict['spanning_width'].append(G_maxs[0] - G_mins[0])
        # info_dict['spanning_height'].append(G_maxs[1] - G_mins[1])
        # info_dict['stretched_path_len'].append(
        #     nx.shortest_path_length(G, source=0, target=len_init, weight='length'))



prop_list = ['N', 's', 'b', 'hull_area', 'tot_len', 'n_edges', 'void_mean', 'void_std']
info_dict = {}
for key in prop_list:
    info_dict[key] = []

ss = np.arange(0, 0.0015, 0.0001)
bs = np.arange(0, 0.15, 0.01)

for s in ss:
    for b in bs:

        s = round(s, 5)
        b = round(b, 5)

        print('now on s =', s, 'b =', b)
        sizes = list(np.arange(100, 300, 5)) + list(np.arange(300, 1010, 50)) + list(np.arange(1010, 2020, 100))
        # sizes = list(np.arange(500, 1100, 100))
        run_ensemble(sizes, s, b)

dataframe = pd.DataFrame(info_dict, columns = prop_list)
pickle.dump(dataframe, open('param_sweep_info.p', "wb"))
