# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2018
# -----------------
#
# * Ronan Hamon r<lastname_AT_protonmail.com>
#
# Description
# -----------
#
# Python implementation of the minimization of cyclic bandwidth problem.
#
# Licence
# -------
# This file is part of pymach.
#
# pymach is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
"""Relabeling of vertices of a graph to follow the network structure.


References
----------
.. [Ham2016] Hamon, R., Borgnat, P., Flandrin, P., & Robardet, C. (2016).
   Relabelling vertices according to the network structure by minimizing the
   cyclic bandwidth sum. Journal of Complex Networks, 4(4), 534-560.

.. moduleauthor:: Ronan Hamon
"""
import heapq as hp
import random

import networkx as nx


def adjust_labeling(labeling, old_marker, new_marker):
    """Adjust a labeling to match `old_marker` with `new_marker` using circular
    invariance of the labeling.

    Parameters
    ----------
    labeling : dict
        A dictionary whose keys are old labels and values new labels.
    old_marker : label
        Label of the marker in the old labeling.
    new_marker : label
        Label of the marker in the new labeling.

    Returns
    -------
    dict
        Adjusted labeling.
    """
    # get the keys and the labels
    old_labels = list(labeling.keys())
    new_labels = list(labeling.values())

    errmsg = "Marker '{}' not in labeling."
    if old_marker not in old_labels:
        raise ValueError(errmsg.format(old_marker))

    if new_marker not in new_labels:
        raise ValueError(errmsg.format(old_marker))

    # get the label of the vertex 0
    shift = new_labels.index(new_marker) - old_labels.index(old_marker)

    # shift all the labels such that the vertex 0 has label 0
    if shift > 0:
        for _ in range(shift):
            new_labels.append(new_labels.pop(0))
    elif shift < 0:
        for _ in range(-shift):
            new_labels.insert(0, new_labels.pop())

    # return the new labeling
    return dict(zip(old_labels, new_labels))


def generate_random_labeling(G, label_attr=None):
    """Return a random labeling for a graph.

    The same set of labels is used to generate the random labeling, only the
    order differs.

    Parameters
    ----------
    G : networkx graph
        Graph to consider.
    label_attr : str or None, optional
        Attribute where the label is stored. If None, use the name of nodes.

    Returns
    -------
    dict
        A labeling with the old labels as keys and the new labels as values.
    """
    if label_attr:
        old_labels = list(nx.get_node_attributes(G, label_attr).values())
    else:
        old_labels = list(G.nodes())

    new_labels = old_labels.copy()
    random.shuffle(new_labels)

    return dict(zip(old_labels, new_labels))


def _cyclic_distance(i, j, n):
    """Return the cyclic distante between two nodes.

    Parameters
    ----------
    i, j : int
        Labels to consider, as integer between 0 and `n_nodes` - 1.
    n : int
        Total number of nodes.

    Return
    ------
    int
        Distance between the two nodes.
    """
    return int(min(abs(i - j), n - abs(i - j)))


def cyclic_bandwidth_sum(G, label_attr=None, weight_attr=None):
    """Return the value of cyclic bandwidth sum of the graph.

    Parameters
    ----------
    G : Networkx graph
        Graph to consider.
    label_attr : str or None, optional
        Attribute where the label is stored. If None, use the name of nodes.
    weight_attr : str or None, optional
        Attribute where the weight is stored. If None, all edges have weight
        equal to 1.

    Returns
    -------
    float
        Value of cyclic bandwidth sum of the graph.
    """

    if label_attr:
        labeling = nx.get_node_attributes(G, label_attr)
    else:
        labeling = {node: node for node in G.nodes()}

    n_nodes = len(labeling)

    cbs = 0
    for u, v in G.edges():
        cbs_uv = _cyclic_distance(labeling[u], labeling[v], n_nodes)

        if weight_attr:
            cbs_uv *= G[u][v][weight_attr]

        cbs += cbs_uv

    return cbs


def jaccard_index(nbors_u, nbors_v, u=None, v=None):
    """Compute the jaccard index between two neighborhood.

    Parameters
    ----------
    nbors_u, nbors_v : set or dict
        Neighborhoods of the nodes to consider. If set, elements are
        neighbors.  If dict, keys are neighbors and values are weight
        associated to each edge.

    Returns
    -------
    float
    """
    if type(nbors_u) == set and type(nbors_v) == set:
        numerator = len(nbors_u.intersection(nbors_v)) + 2.
        denominator = len(nbors_u.union(nbors_v))
    elif type(nbors_u) == dict and type(nbors_u) == dict:
        N1 = 2. * nbors_u[v]
        N2 = sum([min(nbors_u[item], nbors_v[item])
                  for item in
                  set(nbors_u.keys()).intersection(nbors_v.keys())])
        numerator = N1 + N2

        D1 = sum([(nbors_u[item] + nbors_v[item]) / 2
                  for item in
                  set(nbors_u.keys()).intersection(nbors_v.keys())])
        D2 = sum([nbors_u[item]
                  for item in set(nbors_u.keys()).difference(nbors_v.keys())])
        D3 = sum([nbors_v[item]
                  for item in set(nbors_v.keys()).difference(nbors_u.keys())])
        denominator = D1 + D2 + D3

    return numerator / denominator


def _find_paths(G, weight_attr=None, verbose=False):
    """Find a collections of paths in a graph."""

    unvisited_nodes = list(G.nodes())
    paths = list()

    # sort vertices by degree
    degrees = dict(G.degree())
    sources = list(zip(degrees.values(), degrees.keys()))
    hp.heapify(sources)

    # Step 1
    if verbose:
        print('# Step 1 : Find paths\n')

    while sources:

        # select the node with the minimal degree
        source = hp.heappop(sources)[1]

        if source in unvisited_nodes:

            if verbose:
                print("# Path {:d}\n--------".format(len(paths) + 1))

            path = list()
            current_node = source

            # construction of a path from source
            while current_node in unvisited_nodes:

                if verbose:
                    print("Current node: {:d}".format(current_node))

                # add current node in the path
                path.append(current_node)
                # neighborhood of the current node
                nbors_cn = set(G.neighbors(current_node))

                # loop on unvisited neighbors
                jaccards = list()
                for nbor in [item for item in nbors_cn
                             if item in unvisited_nodes]:

                    # if the neighbor has degree 1, add it directly in path
                    if degrees[nbor] == 1:
                        path.append(nbor)
                        unvisited_nodes.remove(nbor)

                        if verbose:
                            print("Neighbor with degree 1: {:d}".format(nbor))

                    # otherwise compute the jaccard index
                    else:
                        # add jaccard index in the heap
                        if weight_attr:
                            nbors_u = {nbor_u:
                                       G[current_node][nbor_u][weight_attr]
                                       for nbor_u in nbors_cn}
                            nbors_v = {nbor_v: G[nbor][nbor_v][weight_attr]
                                       for nbor_v in G.neighbors(nbor)}
                            ji = jaccard_index(
                                nbors_u, nbors_v, current_node, nbor)
                        else:
                            nbors_v = set(G.neighbors(nbor))
                            ji = jaccard_index(nbors_cn, nbors_v)
                        hp.heappush(jaccards, (-ji, nbor))

                # remove the current node from the white nodes
                unvisited_nodes.remove(current_node)

                # select the one with the highest jaccard index
                if jaccards:

                    if verbose:
                        print("    --------")
                        for neighbor in jaccards:
                            message = "    Neighbor {:d} - Index: {:.2f}"
                            print(message.format(neighbor[1], -neighbor[0]))

                    element = hp.heappop(jaccards)
                    current_node = element[1]
                    if verbose:
                        print("Next node: {:d}\n--------".format(current_node))

                if verbose:
                    print("End of the path")

            # insert the path in the list of paths
            hp.heappush(paths, (-len(path), path))

            if verbose:
                print("--------\n")

    if verbose:
        print("\n{:d} paths found\n".format(len(paths)))

    return paths


def _incremental_cbs(G, global_path, chunk, weight_attr):
    """Best index to insert a chunk of nodes in the global path.

    Parameters
    ----------
    G : networkx graph
        Graph to consider.
    weight_attr : str or None
        Attribute where the weight is stored. If None, all edges have weight
        equal to 1.
    global_path : list
        Current global path of the graph.
    chunk : list
        Chunk of labels to insert in the labeling.
    """
    # Initialization of parameters
    p = len(chunk)
    n = len(global_path) + p
    O1 = []
    k = [global_path[0]]
    O2 = global_path[1::]

    # Computation of the CBS for the insertion at index 0
    path_for = chunk + global_path
    path_bac = chunk[::-1] + global_path

    pos_for = dict(zip(path_for, range(n)))
    pos_bac = dict(zip(path_bac, range(n)))

    if weight_attr:

        cbs_for = sum(edge[2][weight_attr] * _cyclic_distance(pos_for[edge[0]],
                                                              pos_for[edge[1]],
                                                              n)
                      for edge in G.edges(chunk, data=True))

        cbs_bac = sum(edge[2][weight_attr] * _cyclic_distance(pos_bac[edge[0]],
                                                              pos_bac[edge[1]],
                                                              n)
                      for edge in G.edges(chunk, data=True))

    else:
        cbs_for = sum(_cyclic_distance(pos_for[edge[0]], pos_for[edge[1]], n)
                      for edge in G.edges(chunk))

        cbs_bac = sum(_cyclic_distance(pos_bac[edge[0]], pos_bac[edge[1]], n)
                      for edge in G.edges(chunk))

    # Initialize the best value of cbs
    best_index = 0
    is_reversed = 0 if cbs_for <= cbs_bac else 1
    best_cbs = min(cbs_for, cbs_bac)

    # Browsing the indices
    for index in range(1, len(global_path) + 1):

        # Different computation between forward and backward
        l0_for = l0_bac = 0
        l1_for = l1_bac = 0
        l2_for = l2_bac = 0

        for node_u in chunk:

            pi_u_for = pos_for[node_u]
            pi_u_bac = pos_for[node_u]
            nbhood_u = set(G.neighbors(node_u))

            # O1
            for node_v in nbhood_u.intersection(O1):
                w = G[node_u][node_v][weight_attr] if weight_attr else 1
                pi_v = pos_for[node_v]
                assert pos_for[node_v] == pos_bac[node_v]

                delta_for = pi_u_for - pi_v
                delta_bac = pi_u_bac - pi_v

                l0_for += w * (-1) ** (delta_for >= (n / 2))
                l0_bac += w * (-1) ** (delta_bac >= (n / 2))

            # 02
            for node_v in nbhood_u.intersection(O2):
                w = G[node_u][node_v][weight_attr] if weight_attr else 1
                pi_v = pos_for[node_v]
                assert pos_for[node_v] == pos_bac[node_v]

                delta_for = pi_v - pi_u_for
                delta_bac = pi_v - pi_u_bac

                l1_for += w * (-1) ** (delta_for <= (n / 2))
                l1_bac += w * (-1) ** (delta_bac <= (n / 2))

        # k
        node_u = k[0]
        pi_u = pos_for[node_u]
        assert pos_for[node_u] == pos_bac[node_u]
        nbhood_u = set(G.neighbors(node_u))

        # path
        for node_v in nbhood_u.intersection(chunk):
            w = G[node_u][node_v][weight_attr] if weight_attr else 1
            pi_v_for = pos_for[node_v]
            pi_v_bac = pos_bac[node_v]

            delta_for = pi_u - pi_v_for
            delta_bac = pi_u - pi_v_bac

            if ((p + 1) - n / 2) <= delta_for <= n / 2:
                l2_for += w * (-2 * delta_for + (p + 1))
            elif delta_for > n / 2:
                l2_for -= w * (n - (p + 1))
            else:
                l2_for += w * (n - (p + 1))

            if ((p + 1) - n / 2) <= delta_bac <= n / 2:
                l2_bac += w * (-2 * delta_bac + (p + 1))
            elif delta_bac > n / 2:
                l2_bac -= w * (n - (p + 1))
            else:
                l2_bac += w * (n - (p + 1))

        # Common computation between forward and backward
        l3 = l4 = 0

        # O1
        for node_v in nbhood_u.intersection(O1):
            w = G[node_u][node_v][weight_attr] if weight_attr else 1
            pi_v = pos_for[node_v]
            assert pos_for[node_v] == pos_bac[node_v]
            a = pi_u - pi_v
            if a <= n / 2:
                l3 -= w * p
            elif a > (n / 2 + p):
                l3 += w * p
            else:
                l3 += w * (2 * a - (n + p))

        # O2
        for node_v in nbhood_u.intersection(O2):
            w = G[node_u][node_v][weight_attr] if weight_attr else 1
            pi_v = pos_for[node_v]
            assert pos_for[node_v] == pos_bac[node_v]
            a = pi_v - pi_u
            if a <= (n / 2 - p):
                l4 += w * p
            elif a >= n / 2:
                l4 -= w * p
            else:
                l4 += w * (-2 * a + (n - p))

        # update cbs
        cbs_for += l0_for + l1_for + l2_for + l3 + l4
        cbs_bac += l0_bac + l1_bac + l2_bac + l3 + l4

        if cbs_for < best_cbs or cbs_bac < best_cbs:
            best_index = index
            is_reversed = 0 if cbs_for <= cbs_bac else 1
            best_cbs = min(cbs_for, cbs_bac)

        # update labeling
        if index < len(global_path):

            pos_for[k[0]] -= p
            pos_bac[k[0]] -= p
            for node in chunk:
                pos_for[node] += 1
                pos_bac[node] += 1

            O1.extend(k)
            k = O2[0:1]
            O2 = O2[1::]

    return best_index, is_reversed


def get_mach_labeling(G, weight_attr=None, verbose=0):
    """Compute a relabeling which follows the structure of the graph.

    Parameters
    ----------
    G : networkx Graph
        Graph to relabel.
    weight_attr : str or None, optional
        Attribute where the weight is stored. If None, all edges have weight
        equal to 1.
    verbose : bool, optional
        Indicates if verbose mode is activated.

    Returns
    -------
    dict
        A mapping with the old labels as keys and the new labels as values.
    """
    # find all paths
    paths = _find_paths(G, weight_attr, verbose)

    # get the largest path
    if verbose:
        print("\n### STEP 2: MERGE PATHS ###\n")

    global_path = hp.heappop(paths)[1]

    while paths:

        # select the longest path in the list of remaining paths
        current_path = hp.heappop(paths)[1]

        # select the corresponding subgraph
        reduced_graph = G.subgraph(global_path + current_path)

        # find the best index to insert
        best_index, is_reversed = _incremental_cbs(
            reduced_graph, global_path, current_path, weight_attr)

        if is_reversed:
            current_path = current_path[::-1]

        global_path = (global_path[0:best_index] +
                       current_path + global_path[best_index::])

    labeling = {node: idn for idn, node in enumerate(global_path)}
    if verbose:
        print("\n\n### Final labeling ###")
        print(labeling)
        print("\nEnd procedure")

    return labeling
