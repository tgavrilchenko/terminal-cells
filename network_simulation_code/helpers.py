import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
#from tree_projection import project_to_plane
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import sys
import os
import itertools
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import seaborn as sns
import math
import pandas as pd
import pickle
from collections import Counter

# number of edges in the graph where all degree 2 nodes
# have been merged is the total number of nodes minus
# the number of degree two nodes
def get_num_coarsened_edges(G):

    degrees = [val for (node, val) in G.degree()]
    count = Counter(degrees)
    return G.number_of_nodes() - count[2] - 1
    #return count[1] + count[3] - 1

def pointillate_hull(hull_coords):
    
    augmented_coords = []

    L = len(hull_coords)
    
    for i in range(L):

        p1 = hull_coords[i]
        p2 = hull_coords[(i + 1) % L]

        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        b = (p2[0]*p1[1] - p1[0]*p2[1])/(p2[0] - p1[0])

        line_len = np.sqrt(np.sum((p1 - p2)**2))

        xs = np.linspace(p1[0], p2[0], int(line_len/5))
        ys = [m*x + b for x in xs]

        augmented_coords += list(zip(xs, ys))

    return augmented_coords

def draw_circle(target, dists_in_hull, M, xpts, ypts, c = 'r'):

    idx = (np.abs(np.asarray(dists_in_hull) - target)).argmin()
    #idx += 50

    M_ind = np.where(M == dists_in_hull[idx])
    xind = M_ind[0][0]
    yind = M_ind[1][0]

    voidx = xpts[xind]
    voidy = ypts[yind]
    voidr = M[xind, yind]

    # N = 100
    # circlex = [voidx + voidr * np.cos(alpha) for alpha in np.linspace(0, 2 * np.pi, N)]
    # circley = [voidy + voidr * np.sin(alpha) for alpha in np.linspace(0, 2 * np.pi, N)]
    # # inds = [i for i in range(N) if in_hull([xs[i], ys[i]], np.array(hull_coords))]
    #
    # for i in range(len(circlex) - 1):
    #     plt.plot([circlex[i], circlex[i + 1]], [circley[i], circley[i + 1]], color=c, linewidth=1)

    return plt.Circle((voidx, voidy), voidr, color = c, alpha = 0.5, ec = None)

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def compute_void(G, ax = None, plotting = False):

    coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])

    coords = unique_rows(coords)

    hull = ConvexHull(coords)

    # print(G.number_of_nodes())
    # print(coords)
    # print('here:', unique_rows(coords))
    
    hull_coords = np.array(coords[hull.vertices])

    additional_coords = pointillate_hull(hull_coords)

    #print(additional_coords)

    if len(additional_coords) > 0:

        coords = np.concatenate((coords, additional_coords))
    #print(len(coords), len(additional_coords))

    # plt.clf()
    # for p in coords:
    #     plt.scatter(p[0], p[1], color = 'k', s = 1)
    # plt.show()
    # sys.exit()

    N = 301

    xmin = np.min(hull_coords[:, 0])
    xmax = np.max(hull_coords[:, 0])
    ymin = np.min(hull_coords[:, 1])
    ymax = np.max(hull_coords[:, 1])


    #print(xmin, xmax, ymin, ymax)

    xpts = np.linspace(xmin, xmax, N)
    ypts = np.linspace(ymin, ymax, N)
    grid = list(itertools.product(xpts, ypts))

    #print('size:', len(grid), len(coords))

    closest_dist = np.min(cdist(grid, coords, 'euclidean'), axis=1)

    result = in_hull(grid, np.array(hull_coords))

    #print(N*N, len(np.where(result== True)[0]))
    #sys.exit()

    # only set points outside of the convex hull to 0
    
    dists_in_hull = closest_dist[np.where(result == True)[0]]
    closest_dist[np.where(result != True)[0]] = 0

    M = closest_dist.reshape((N, N))

    ## Plotting
    plt.plot(hull_coords[:, 0], hull_coords[:, 1], 'k--', lw=1, alpha=0.5)
    plt.plot([hull_coords[0][0], hull_coords[-1][0]],
             [hull_coords[0][1], hull_coords[-1][1]], 'k--', lw=1, alpha=0.5)

    # ##plot max void
    #
    # circle = draw_circle(np.max(dists_in_hull), dists_in_hull, M, xpts, ypts, c = 'r')
    # ax.add_patch(circle)
    #
    # print(np.mean(dists_in_hull), np.std(dists_in_hull))
    #
    # target = np.mean(dists_in_hull) + .012*np.std(dists_in_hull)
    #
    # circle = draw_circle(target, dists_in_hull, M, xpts, ypts, c = 'b')
    # ax.add_patch(circle)
    #
    #
    #
    # ##plot average void
    #
    # avg_void = np.mean(dists_in_hull)
    # idx = (np.abs(np.asarray(dists_in_hull) - avg_void)).argmin()
    # target = dists_in_hull[idx]
    #
    # print(target)
    #
    # M_ind = np.where(M == target)
    # print(M_ind[0][0], M_ind[1][0])
    

    if plotting:

        colors = [(1, 1, 1), (0, 0, 0.60)]  # first color is black, last is blue
        #colors = [(0, 0, 0.7), (1, 1, 1)]  # reverse colors
        cm = LinearSegmentedColormap.from_list("Custom", colors, N=100)

        fig, ax = plt.subplots(figsize=(4, 4))
        X1, Y1 = np.mgrid[0:1:complex(N), 0:1:complex(N)]
        im = ax.pcolormesh(X1, Y1, M, shading='auto', cmap=cm)  # , vmin=0, vmax=0)

        #plt.imshow(M, interpolation = 'none', cmap = cm)


        # plt.hist(closest_dist, bins = np.linspace(0, 16, 50))
        #plt.hist(dists_in_hull, bins = 30)

        for e in G.edges(data=True):
            c0 = (G.nodes[e[0]]['coords'][:2] - (xmin, ymin))/(xmax-xmin , ymax- ymin)
            c1 = (G.nodes[e[1]]['coords'][:2] - (xmin, ymin))/(xmax-xmin , ymax- ymin)

            # c0 = G.nodes[e[0]]['coords'][:2]
            # c1 = G.nodes[e[1]]['coords'][:2]

            rad = 0.5*(G.nodes[e[0]]['radius']+G.nodes[e[1]]['radius'])
            plt.plot([c0[0], c1[0]], [c0[1], c1[1]], linewidth = rad, color = 'k')

        # #hull_coords = (hull_coords - (xmin, ymin))/(xmax-xmin , ymax- ymin)
        # plt.plot(hull_coords[:, 0], hull_coords[:, 1], 'k--', lw=1, alpha=0.5)
        # plt.plot([hull_coords[0][0], hull_coords[-1][0]],
        #          [hull_coords[0][1], hull_coords[-1][1]], 'k--', lw=1, alpha=0.5)

        fig.colorbar(im)
        # plt.axis('off')
        plt.savefig('.png')

    return dists_in_hull

def edge_lengths(H3D, H):
    # H3D: initial 3D tree
    # H: projected 2D tree

    # give H edge attributes
    for e in H.edges():
        c1 = H.nodes[e[0]]['coords'][:2]
        c2 = H.nodes[e[1]]['coords'][:2]
        H[e[0]][e[1]]['elen'] = np.sqrt(sum(c1 - c2) ** 2)

    # first, simplify H into G where all deg 2 nodes are removed
    # then, for all (e1, e2) in G, compute the min distance between
    # these nodes in H, giving the true (projected) edge length

    G = H.copy()

    special = [n for n in G.nodes() if G.degree(n) != 2]

    while G.number_of_nodes() > len(special):
        for n in G.nodes():
            if G.degree(n) == 2:
                neibs = list(G.neighbors(n))
                G.add_edge(neibs[0], neibs[1])
                G.remove_node(n)

                break

    e0 = 1
    e1 = list(G.neighbors(1))[0]
    first_e_len = nx.shortest_path_length(H, source=e0,
                                       target=e1, weight='elen')
    # c1 = G.nodes[e0]['coords'][:2]
    # c2 = G.nodes[e1]['coords'][:2]
    # print(true_len, np.sqrt(sum(c1 - c2) ** 2))

    len_tips = 0
    tip_list = []

    for e in G.edges():
        #print(e[0], e[1])
        elen = nx.shortest_path_length(H, source=e[0],
                                       target=e[1], weight='elen')
        G[e[0]][e[1]]['elen'] = elen

        if(e[0] != 1 and e[1] != 1) and \
                    (len(list(G.neighbors(e[0]))) == 1 or 
                     len(list(G.neighbors(e[1]))) == 1):
            
            len_tips += elen
            if len(list(G.neighbors(e[0]))) == 1:
                tip_list.append(e[0])
            elif len(list(G.neighbors(e[1]))) == 1:
                tip_list.append(e[1])

    # find the number of generations in the tree
    # go through the tip list, find number of hops to the root (node 1)
    # the max of all values is the order.

    hop_dists = [nx.shortest_path_length(G, source=n, target=1) for n in tip_list]

    if len(hop_dists) == 0:
        hop_dists = [1]

    max_path_len = np.max(hop_dists)

    #print('all tip-root paths:', hop_dists, max_path_len)

    #num_nodes = G.number_of_nodes()
    avg_e_len = np.mean(list(nx.get_edge_attributes(G, 'elen').values()))

    return first_e_len, avg_e_len, len_tips, max_path_len

def get_children(G, n):
    neighbors = list(G.neighbors(n))

    return [m for m in neighbors if n < m]

def strahler_ordering(H):
    # set all degree 1 nodes in H that are not the root (n=1) to S number 1
    # add all neighbors of S labeled nodes to the iteration list
    # proceed until iteration list is empty

    nx.set_node_attributes(H, 0, 'strahler')

    for n in H.nodes():
        if len(list(H.neighbors(n))) == 1 and n != 1:
            H.nodes[n]['strahler'] = 1

    # print(nx.get_node_attributes(H, 'strahler'))

    existUnlabeled_nodes = True

    while existUnlabeled_nodes:

        strahler_dict = nx.get_node_attributes(H, 'strahler')

        unlabeled = [k for k, v in strahler_dict.items() if v == 0]

        if len(unlabeled) == 0:
            existUnlabeled_nodes = False
        else:

            for n in unlabeled:
                children = get_children(H, n)

                if len(children) == 2:
                    S1 = H.nodes[children[0]]['strahler']
                    S2 = H.nodes[children[1]]['strahler']
                    # if one child is unlabeled, do not label
                    if S1 > 0 and S2 > 0:
                        if S1 == S2:
                            H.nodes[n]['strahler'] = S1 + 1
                        else:
                            H.nodes[n]['strahler'] = max([S1, S2])

                        #print('node', n, 'strahler labeled', H.nodes[n]['strahler'])
                elif len(children) == 1:
                    S1 = H.nodes[children[0]]['strahler']
                    if S1 > 0:
                        H.nodes[n]['strahler'] = S1

    # strahler_dict = nx.get_node_attributes(H, 'strahler')
    # print(strahler_dict)
    # sys.exit()

    return H

def get_hierarchy(G, H):
    # G: projected 2D tree with geometry info (use for edge lengths)
    # H: projected 2D tree with deg 2 nodes removed (use for Strahler ordering)
    # node labels are preserved between trees

    # return: list of Strahler orders and lengths for each edge
    # the original tree G with Strahler edge labels

    ## give G edge attributes
    for e in G.edges():
        c1 = G.nodes[e[0]]['coords'][:2]
        c2 = G.nodes[e[1]]['coords'][:2]
        G[e[0]][e[1]]['elen'] = np.sqrt(sum(c1 - c2) ** 2)


    H = strahler_ordering(H)
    
    all_strahler_nums = nx.get_node_attributes(H, 'strahler').values()
    max_order = max(all_strahler_nums) #+ 1
    for n in H.nodes():
        H.nodes[n]['strahler'] = max_order - H.nodes[n]['strahler']

    nx.set_node_attributes(G, 0, 'strahler')
    G.nodes[1]['strahler'] = H.nodes[1]['strahler']

    #print('nodes:', H.number_of_nodes(), G.number_of_nodes())

    strahler_list = []
    elen_list = []

    for e in H.edges():

        n1 = min(e)
        n2 = max(e)

        elen = nx.shortest_path_length(G, source=n2, target=n1, weight='elen')
        
        path = nx.shortest_path(G, source=n2, target=n1)
        #print('path:', path, H.nodes[n2]['strahler'])

        for i in range(len(path) -1):
            G.edges[path[i], path[i+1]]['strahler'] = H.nodes[n2]['strahler']

        # elen2 = 0
        # i = 1
        # for n in path:
        #     G.nodes[n]['strahler'] = H.nodes[n2]['strahler']
        #     elen2 += G[path[i]][path[i-1]]['elen']
        #     i += 1
        # print(elen, elen2)
            

        #print(e[0], e[1], e_strahler, elen)
        strahler_list.append(H.nodes[n2]['strahler'])
        elen_list.append(elen)

    return strahler_list, elen_list, G

def get_topological_hierarchy(G, H):
    # G: projected 2D tree with geometry info (use for edge lengths)
    # H: projected 2D tree with deg 2 nodes removed (use for Strahler ordering)
    # node labels are preserved between trees

    # return: list of Strahler orders and lengths for each edge
    # the original tree G with Strahler edge labels

    ## give G edge attributes
    for e in G.edges():
        c1 = G.nodes[e[0]]['coords'][:2]
        c2 = G.nodes[e[1]]['coords'][:2]
        G[e[0]][e[1]]['elen'] = np.sqrt(sum(c1 - c2) ** 2)

    # H = strahler_ordering(H)
    #
    # all_strahler_nums = nx.get_node_attributes(H, 'strahler').values()
    # max_order = max(all_strahler_nums)  # + 1
    for n in H.nodes():
        H.nodes[n]['order'] = nx.shortest_path_length(H, source=n, target=1)

    nx.set_node_attributes(G, 0, 'order')
    G.nodes[1]['order'] = H.nodes[1]['order']

    # print('nodes:', H.number_of_nodes(), G.number_of_nodes())

    order_list = []
    elen_list = []

    for e in H.edges():

        n1 = min(e)
        n2 = max(e)

        elen = nx.shortest_path_length(G, source=n2, target=n1, weight='elen')

        path = nx.shortest_path(G, source=n2, target=n1)
        # print('path:', path, H.nodes[n2]['strahler'])

        for i in range(len(path) - 1):
            G.edges[path[i], path[i + 1]]['order'] = H.nodes[n2]['order']

        # elen2 = 0
        # i = 1
        # for n in path:
        #     G.nodes[n]['strahler'] = H.nodes[n2]['strahler']
        #     elen2 += G[path[i]][path[i-1]]['elen']
        #     i += 1
        # print(elen, elen2)

        # print(e[0], e[1], e_strahler, elen)
        order_list.append(H.nodes[n2]['order'])
        elen_list.append(elen)

    return order_list, elen_list, G

# def total_edge_length(G):
# 
#     tot_len = 0
#     #len_tips = 0
# 
#     for e in G.edges():
#         e_len = np.sqrt(np.sum((G.nodes[e[0]]['coords'] -
#                                    G.nodes[e[1]]['coords']) ** 2))
#         tot_len += e_len
# 
#         # if (e[0]!= 1 and e[1]!= 1) and \
#         #         (len(list(G.neighbors(e[0]))) == 1 or len(list(G.neighbors(e[1]))) == 1):
#         #     len_tips += e_len
# 
#     return tot_len#, len_tips

# G: graph with edge coords
def total_edge_length(G):
    return np.sum(list(nx.get_edge_attributes(G, 'length').values()))

# G: graph with edge coords
def convex_hull_area(G):
    return ConvexHull(list(nx.get_node_attributes(G, 'coords').values())).volume

def number_of_branches(G):

    all_degrees = [val for (node, val) in G.degree()]
    n1 = all_degrees.count(1)
    n3 = all_degrees.count(3)
    return n1 + n3 - 1


def get_coarsened_edge_lengths(G, H):
    # G: coarsened tree with geometry for plotting
    # H: G with all deg 2 nodes removed
    # node labels are preserved between trees

    # return: K but with edges labeled by the length of the
    # full branch in H they belong to, with distances from node
    # coordinates in G

    ## give G edge attributes
    for e in G.edges():
        c1 = np.array(G.nodes[e[0]]['coords'][:2])
        c2 = np.array(G.nodes[e[1]]['coords'][:2])
        G[e[0]][e[1]]['length'] = np.sqrt(sum(c1 - c2)** 2)

    for e in H.edges():

        elen = nx.shortest_path_length(G, source=e[0], target=e[1], weight='length')
        path = nx.shortest_path(G, source=e[0], target=e[1])

        c1 = np.array(H.nodes[e[0]]['coords'][:2])
        c2 = np.array(H.nodes[e[1]]['coords'][:2])
        H_len = np.sqrt(sum(c1 - c2)** 2)
        
        H.edges[e[0], e[1]]['global_elen'] = H_len

        for i in range(len(path) - 1):
            G.edges[path[i], path[i + 1]]['global_elen'] = elen

    return G

def branch_split_labeling(G):

    label_dict = {}

    # list of split nodes, including node 1
    for n in range(G.number_of_nodes()+1)[1:]:
        if n == 1:
            label_dict[n] = n
        # elif G.degree(n) == 3:
        #     label_dict[n] = n
        else:
            m = np.min(list(G.neighbors(n)))
            if G.degree(m) == 3:
                label_dict[n] = n
            else:
                label_dict[n] = label_dict[m]

    return label_dict

def get_angle(p0, p1, p2):

    v0 = np.array(p1) - np.array(p0)
    v1 = np.array(p1) - np.array(p2)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return abs(np.degrees(angle))

# create subgraph of straight edges
def coarsen_graph(G, max_separation = math.inf, relabel_opt = True):

    # first, simplify G into H where all deg 2 nodes are removed

    H = G.copy()

    special = [n for n in H.nodes() if H.degree(n) != 2]

    while H.number_of_nodes() > len(special):
        for n in H.nodes():
            if H.degree(n) == 2:
                neibs = list(H.neighbors(n))
                H.add_edge(neibs[0], neibs[1])
                H.remove_node(n)

                break
    if max_separation == math.inf:
        S = H

    else:

        new_edges = []
        new_nodes = []

        # for each edge in the fully simplified graph, find the path
        # in the complete graph and remove every Nth node (N = max_separation)

        for e in H.edges():

            path = nx.shortest_path(G, e[0], e[1])

            # print(path)
            # print(e[0], e[1])
            shortened_path = path[::max_separation]
            # print(shortened_path)

            if e[1] not in shortened_path:
                shortened_path.append(e[1])

            new_nodes = new_nodes + shortened_path

            for i in range(len(shortened_path)-1):
                new_edges.append((shortened_path[i], shortened_path[i+1]))

            #print(new_edges)

        S = G.subgraph(list(set(new_nodes))).copy()

        for i in range(len(new_edges)):
            e = new_edges[i]
            S.add_edge(e[0], e[1])

    root = list(S.nodes())[0]

    # relabel nodes to be strictly increasing with increasing distance from root node
    paths = nx.shortest_path(S, source=root)
    paths = [(k, len(paths[k])) for k in paths.keys()]
    mapping = {paths[i][0]: i + 1 for i in range(len(paths))}

    if relabel_opt:
        S = nx.relabel_nodes(S, mapping)

    return S

# G: projected 2D tree
def branching_angles(G, coarsening, savename, plotting = False):

    S = coarsen_graph(G, coarsening)
    #S = G

    angles = []
    angle_ratios = []
    angles_3 = []

    split_labels = branch_split_labeling(S)

    angle_dict = {}

    for n in S.nodes():

        if S.degree(n) == 3:
            #plt.text(c[0] + 0.02, c[1] + 0.02, str(n), color='k', size=5)

            neibs = sorted(list(S.neighbors(n)))

            p = S.nodes[n]['coords'][:2]
            p0 = S.nodes[neibs[0]]['coords'][:2]
            p1 = S.nodes[neibs[1]]['coords'][:2]
            p2 = S.nodes[neibs[2]]['coords'][:2]

            a1 = get_angle(p0, p, p1)
            a2 = get_angle(p0, p, p2)
            a3 = get_angle(p1, p, p2)

            #print(max(a1, a2, a3))

            if round(a1+a2+a3, 5) != 360:
                if a1 == max(a1, a2, a3): a1 = 360 - a1
                elif a2 == max(a1, a2, a3): a2 = 360 - a2
                elif a3 == max(a1, a2, a3): a3 = 360 - a3

            # if n == 18:
            #     print(a1, a2, a3)
            #     print(a1+a2+a3)
            #     sys.exit()

            if round(a1+a2+a3, 5) != 360:
                print('problem here')

            angle_dict[neibs[1]] = a1
            angle_dict[neibs[2]] = a2

            angles.append(int(a1))
            angles.append(int(a2))
            angle_ratios.append(max(a1, a2) / min(a1, a2))
            angles_3.append(int(a3))

    for e in S.edges():
        n = max(e[0], e[1])
        m = min(e[0], e[1])

        if split_labels[n] in angle_dict:
            angle = angle_dict[split_labels[n]]

            S.edges[n, m]['angle'] = angle_dict[split_labels[n]]

    if plotting:

        fig, ax = plt.subplots(figsize=(5, 4))

        # for n in S.nodes():
        #     c = S.nodes[n]['coords']
            # plt.text(c[0] + 0.5, c[1] + 0.06, str(split_labels[n]), color='b', size=3)

            # if split_labels[n] in angle_dict:
            #     plt.text(c[0] + 0.5, c[1] + 0.06, str(int(angle_dict[split_labels[n]])), color='b', size=3)

        palette = sns.color_palette("rocket", as_cmap=True)
        palette = sns.color_palette("icefire", as_cmap=True)
        palette = sns.color_palette("hls", 10, as_cmap=True)
        palette = sns.color_palette("rocket_r", as_cmap=True)

        print("number of edges:", S.number_of_edges())
        
        for n in G.nodes():
            #plt.scatter(G.nodes[n]['coords'][0], G.nodes[n]['coords'][1], s = 4, color = 'k')
            plt.scatter(G.nodes[n]['coords'][0], G.nodes[n]['coords'][1], s = 4*G.nodes[n]['radius'], color = 'k')

        # for n in S.nodes():
        #     plt.scatter(S.nodes[n]['coords'][0], S.nodes[n]['coords'][1], s = 4, color = 'r')
        #     c = S.nodes[n]['coords']
        #     plt.text(c[0] + 0.5, c[1] + 0.06, str(n), color='b', size=3)

        for e in S.edges():
            n = max(e[0], e[1])
            c = 'gray'

            if split_labels[n] in angle_dict:
                
                angle = angle_dict[split_labels[n]]
                
                c = palette(angle/360)
                # cn = S.nodes[n]['coords']
                #print(S.nodes[n]['radius'])
                #plt.text(cn[0] + 0.5, cn[1] + 0.06, int(angle), color='b', size=3)
                
                # if angle > 70 and angle < 110:
                #     c = 'r'
                # elif angle > 170 and angle < 190:
                #     c = 'b'

            c0 = S.nodes[e[0]]['coords']
            c1 = S.nodes[e[1]]['coords']
            plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color = c, linewidth = 1)
            #plt.plot([c0[0], c1[0]], [c0[1], c1[1]], color = c, linewidth = S.nodes[n]['radius'])

        sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(0, 360))
        plt.colorbar(sm)#, label='edge branching angle')
        plt.axis('scaled')
        plt.savefig('projections_with_angles/' + savename + '_' + str(coarsening) + '.pdf')
        #plt.savefig('angles_2/' + savename + '_' + str(coarsening) + '.pdf')
        plt.clf()

    return S, angles, angle_ratios, angles_3

# intersection of circle and line segment
# c: center of the circle, r: radius of the circle
# a: point inside circle, b: point outside circle
# return: length of line segment ab that is inside the circle
def partial_line(c, r, n, m):
    if n[0] != m[0]:

        slope = (n[1] - m[1]) / (n[0] - m[0])
        incept = (n[0] * m[1] - n[1] * m[0]) / (n[0] - m[0])

        A = 1 + slope ** 2
        B = -2 * c[0] - 2 * slope * (c[1] - incept)
        C = c[0] ** 2 + (c[1] - incept) ** 2 - r ** 2

        disc = B ** 2 - 4 * A * C

        if disc < 0:
            return 0

        else:

            p1x = (-B + np.sqrt(disc)) / (2 * A)
            p1y = slope * p1x + incept

            p2x = (-B - np.sqrt(disc)) / (2 * A)
            p2y = slope * p2x + incept

            d1_sq = (m[0] - p1x) ** 2 + (m[1] - p1y) ** 2
            d2_sq = (m[0] - p2x) ** 2 + (m[1] - p2y) ** 2

            p = np.array((p1x, p1y))

            # print(slope, incept, p)

            if d2_sq < d1_sq:
                p = np.array((p2x, p2y))


    else:

        py = np.sqrt(r ** 2 - (n[0] - c[0]) ** 2)

        # m is above n, and the circle intersection point must be in the middle
        if n[1] < m[1]:
            p = (n[0], c[1] + py)
        # other case
        else:
            # print('other case')
            p = (n[0], c[1] - py)

    # visualize(c, r, n, m, p)
    return np.sqrt(np.sum((n - p) ** 2))