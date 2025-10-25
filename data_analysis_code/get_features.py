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

def pointillate_hull(hull_coords):
    
    augmented_coords = []

    L = len(hull_coords)
    
    for i in range(L):

        p1 = hull_coords[i]
        p2 = hull_coords[(i + 1) % L]

        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        b = (p2[0]*p1[1] - p1[0]*p2[1])/(p2[0] - p1[0])

        line_len = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

def compute_void(G, ax = None, plotting = False):

    coords = np.array([i[:2] for i in nx.get_node_attributes(G, 'coords').values()])
    hull = ConvexHull(coords)
    
    hull_coords = np.array(coords[hull.vertices])

    additional_coords = pointillate_hull(hull_coords)

    coords = np.concatenate((coords, additional_coords))
    print('CH coords', len(coords), len(additional_coords))

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

    print(xmin, xmax, ymin, ymax)

    xpts = np.linspace(xmin, xmax, N)
    ypts = np.linspace(ymin, ymax, N)
    grid = list(itertools.product(xpts, ypts))

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
    target = np.mean(dists_in_hull) + .012*np.std(dists_in_hull)
    circle = draw_circle(target, dists_in_hull, M, xpts, ypts, c = 'b')
    ax.add_patch(circle)

    sorted_dists = sorted(dists_in_hull)[40000:60000]

    print(len(sorted_dists))

    for d in sorted_dists[::3200]:

        #target = np.mean(dists_in_hull) - .012 * np.std(dists_in_hull)
        target = d
        circle = draw_circle(target, dists_in_hull, M, xpts, ypts, c='b')
        ax.add_patch(circle)

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
        H[e[0]][e[1]]['elen'] = np.sqrt(sum((c1 - c2) ** 2))

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
        G[e[0]][e[1]]['elen'] = np.sqrt(sum((c1 - c2) ** 2))


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
        G[e[0]][e[1]]['elen'] = np.sqrt(sum((c1 - c2) ** 2))

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


# G: graph with edge coords
def total_edge_length(G):
    return np.sum(list(nx.get_edge_attributes(G, 'length').values()))

# G: graph with edge coords
def convex_hull_area(G):
    return ConvexHull(list(nx.get_node_attributes(G, 'coords').values())).volume


def get_coarsened_edge_lengths(G, H):
    # G: coarsened tree with geometry for plotting
    # H: G with all deg 2 nodes removed
    # node labels are preserved between trees
    

    # return: K but with edges labeled by the length of the
    # full branch in H they belong to, with distances from node
    # coordinates in G


    ## give G edge attributes
    # for e in G.edges():
    #     c1 = G.nodes[e[0]]['coords'][:2]
    #     c2 = G.nodes[e[1]]['coords'][:2]
    #     G[e[0]][e[1]]['elen'] = np.sqrt(sum((c1 - c2)** 2))

    for e in H.edges():

        elen = nx.shortest_path_length(G, source=e[0], target=e[1], weight='elen')
        path = nx.shortest_path(G, source=e[0], target=e[1])

        c1 = H.nodes[e[0]]['coords'][:2]
        c2 = H.nodes[e[1]]['coords'][:2]
        H_len = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
        H.edges[e[0], e[1]]['global_elen'] = H_len

        for i in range(len(path) - 1):
            G.edges[path[i], path[i + 1]]['global_elen'] = elen

    return G

# create subgraph of straight edges
def coarsen_graph(G, max_separation = math.inf, relabel_opt = True, root = 1):

    #print(G.nodes())

    # first, simplify G into H where all deg 2 nodes are removed

    H = G.copy()

    special = [n for n in H.nodes() if H.degree(n) != 2 or n == root]

    while H.number_of_nodes() > len(special):
        for n in H.nodes():
            if H.degree(n) == 2 and n != root:
                neibs = list(H.neighbors(n))
                H.add_edge(neibs[0], neibs[1])
                H.remove_node(n)

                break
    if max_separation == math.inf:
        S = H

    # take every max_separationth node in each path
    # else:
    # 
    #     new_edges = []
    #     new_nodes = []
    # 
    #     # for each edge in the fully simplified graph, find the path
    #     # in the complete graph and remove every Nth node (N = max_separation)
    # 
    #     for e in H.edges():
    # 
    #         path = nx.shortest_path(G, e[0], e[1])
    # 
    #         # print(path)
    #         # print(e[0], e[1])
    #         shortened_path = path[::max_separation]
    #         # print(shortened_path)
    # 
    #         if e[1] not in shortened_path:
    #             shortened_path.append(e[1])
    # 
    #         new_nodes = new_nodes + shortened_path
    # 
    #         for i in range(len(shortened_path)-1):
    #             new_edges.append((shortened_path[i], shortened_path[i+1]))
    # 
    #         #print(new_edges)
    # 
    #     S = G.subgraph(list(set(new_nodes))).copy()
    # 
    #     for i in range(len(new_edges)):
    #         e = new_edges[i]
    #         S.add_edge(e[0], e[1])
    
    # take nodes so that segments are split into max-separation-long pieces
    else:

        new_edges = []
        new_nodes = []

        # for each edge in the fully simplified graph, find the path
        # in the complete graph and remove every Nth node (N = max_separation)

        for e in H.edges():

            path = nx.shortest_path(G, e[0], e[1])

            shortened_path = [path[0]]

            n = e[0]

            for i in range(len(path)):

                m = path[i]

                c0 = G.nodes[n]['coords']
                c1 = G.nodes[m]['coords']
                d = np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)

                if d > max_separation:
                    shortened_path.append(m)
                    n = m

            if shortened_path[-1] != m:
                shortened_path.append(m)


            new_nodes = new_nodes + shortened_path

            for i in range(len(shortened_path)-1):
                new_edges.append((shortened_path[i], shortened_path[i+1]))

            #print(new_edges)

        S = G.subgraph(list(set(new_nodes))).copy()

        for i in range(len(new_edges)):
            e = new_edges[i]
            S.add_edge(e[0], e[1])
    

    # relabel nodes to be strictly increasing with increasing distance from root node
    paths = nx.shortest_path(S, source=root)
    paths = [(k, len(paths[k])) for k in paths.keys()]
    mapping = {paths[i][0]: i + 1 for i in range(len(paths))}

    if relabel_opt:
        S = nx.relabel_nodes(S, mapping)

    return S