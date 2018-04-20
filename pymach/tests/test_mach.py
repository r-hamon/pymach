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
"""Test of the module :mod:`pymach.mach`."""
import pytest

import networkx as nx

from pymach.mach import _cyclic_distance
from pymach.mach import adjust_labeling
from pymach.mach import cyclic_bandwidth_sum
from pymach.mach import get_mach_labeling
from pymach.mach import generate_random_labeling

from .utils import cbs_path
from .utils import cbs_cycle
from .utils import cbs_wheel
from .utils import cbs_power_graph_cycles
from .utils import cbs_complete_bipartite_graph


@pytest.fixture(scope='class')
def get_data(request):
    # define a simple graph
    request.cls.g = nx.Graph()
    request.cls.g.add_edge('a', 'b', weight=2)
    request.cls.g.add_edge('a', 'c', weight=10)
    request.cls.g.add_edge('a', 'd', weight=1)
    request.cls.g.add_edge('b', 'd', weight=1)
    request.cls.g.add_edge('b', 'e', weight=2)
    request.cls.g.add_edge('c', 'd', weight=5)

    # define integer labels
    request.cls.l_abc = dict(zip(['a', 'b', 'c', 'd', 'e'], range(5)))
    nx.set_node_attributes(request.cls.g, request.cls.l_abc, 'label')


@pytest.mark.usefixtures('get_data')
class TestMach:

    def test_adjust_labeling(self):

        # No shift
        adj_labeling = adjust_labeling(self.l_abc, 'a', 0)
        exp_adj_labeling = dict(zip(['a', 'b', 'c', 'd', 'e'], range(5)))
        assert adj_labeling == exp_adj_labeling

        # Forward shift
        adj_labeling = adjust_labeling(self.l_abc, 'b', 0)
        exp_adj_labeling = dict(zip(['b', 'c', 'd', 'e', 'a'], range(5)))
        assert adj_labeling == exp_adj_labeling

        # Backward shift
        adj_labeling = adjust_labeling(self.l_abc, 'a', 4)
        exp_adj_labeling = dict(zip(['b', 'c', 'd', 'e', 'a'], range(5)))
        assert adj_labeling == exp_adj_labeling

        match = "Marker '\w+' not in labeling."
        with pytest.raises(ValueError, match=match):
            adjust_labeling(self.l_abc, 'a', 'b')

        with pytest.raises(ValueError, match=match):
            adjust_labeling(self.l_abc, 0, 1)

    def test_generate_random_labeling(self):

        rdm_labeling = generate_random_labeling(self.g)
        assert isinstance(rdm_labeling, dict)
        assert set(rdm_labeling.values()) == set(['a', 'b', 'c', 'd', 'e'])

        rdm_labeling = generate_random_labeling(self.g, label_attr='label')
        assert isinstance(rdm_labeling, dict)
        assert set(rdm_labeling.values()) == set(range(5))

    def test_cyclic_distance(self):

        n_nodes = 100

        assert _cyclic_distance(0, 1, n_nodes) == 1
        assert _cyclic_distance(0, n_nodes // 4, n_nodes) == n_nodes // 4
        assert _cyclic_distance(0, n_nodes // 2 - 1,
                                n_nodes) == n_nodes // 2 - 1
        assert _cyclic_distance(0, n_nodes // 2, n_nodes) == n_nodes // 2
        assert _cyclic_distance(0, n_nodes // 2 + 1,
                                n_nodes) == n_nodes // 2 - 1
        assert _cyclic_distance(0, n_nodes - n_nodes //
                                4, n_nodes) == n_nodes // 4
        assert _cyclic_distance(0, n_nodes - 1, n_nodes) == 1

        n_nodes = 101

        assert _cyclic_distance(0, 1, n_nodes) == 1
        assert _cyclic_distance(0, n_nodes // 4, n_nodes) == n_nodes // 4
        assert _cyclic_distance(0, n_nodes // 2, n_nodes) == n_nodes // 2
        assert _cyclic_distance(0, n_nodes // 2 + 1, n_nodes) == n_nodes // 2
        assert _cyclic_distance(0, n_nodes - n_nodes //
                                4, n_nodes) == n_nodes // 4
        assert _cyclic_distance(0, n_nodes - 1, n_nodes) == 1

    def test_cyclic_bandwidth_sum(self):

        assert cyclic_bandwidth_sum(self.g, label_attr='label') == 10
        assert cyclic_bandwidth_sum(
            self.g, label_attr='label', weight_attr='weight') == 35

        with pytest.raises(TypeError, match=''):
            assert cyclic_bandwidth_sum(self.g) == 10

    def test_get_mach_labeling(self):

        mach_labeling = get_mach_labeling(self.g)
        assert isinstance(mach_labeling, dict)
        assert set(mach_labeling.keys()) == set(['a', 'b', 'c', 'd', 'e'])

        mach_labeling = get_mach_labeling(self.g)
        assert isinstance(mach_labeling, dict)
        assert set(mach_labeling.keys()) == set(['a', 'b', 'c', 'd', 'e'])


GRAPH_MODELS = {'path': (nx.path_graph, cbs_path),
                'cycle': (nx.cycle_graph, cbs_cycle),
                'wheel': (nx.wheel_graph, cbs_wheel),
                'pgc2': (lambda n: nx.watts_strogatz_graph(n, 4, 0.0),
                         lambda n: cbs_power_graph_cycles(n, 2)),
                'pgc10': (lambda n: nx.watts_strogatz_graph(n, 20, 0.0),
                          lambda n: cbs_power_graph_cycles(n, 10)),
                'cbp1': (lambda n: nx.complete_bipartite_graph(int(n*0.5),
                                                               int(n*0.5)),
                         lambda n: cbs_complete_bipartite_graph(int(n*0.5),
                                                                int(n*0.5))),
                'cbp3': (lambda n: nx.complete_bipartite_graph(int(n*0.25),
                                                               int(n*0.75)),
                         lambda n: cbs_complete_bipartite_graph(int(n*0.25),
                                                                int(n*0.75))),
                'cbp7': (lambda n: nx.complete_bipartite_graph(int(n*0.875),
                                                               int(n*0.125)),
                         lambda n: cbs_complete_bipartite_graph(int(n*0.875),
                                                                int(n*0.125)))}


class TestMachGraphModel:

    def test_standard_unweighted_graphs(self):

        for name, model in GRAPH_MODELS.items():
            for n_nodes in [50, 75, 100]:

                print('{} with {} nodes'.format(name, n_nodes))

                g = model[0](n_nodes)
                cbs_th = model[1](n_nodes)

                # shuffle labels
                random_labeling = generate_random_labeling(g)
                g = nx.relabel_nodes(g, random_labeling)

                assert cyclic_bandwidth_sum(g) != cbs_th

                # relabel graph according to cbs
                mach_labeling = get_mach_labeling(g)
                g = nx.relabel_nodes(g, mach_labeling)

                assert cyclic_bandwidth_sum(g) == cbs_th

    def test_standard_weighted_graphs(self):

        for name, model in GRAPH_MODELS.items():
            for n_nodes in [50, 100]:

                print('{} with {} nodes'.format(name, n_nodes))

                g = model[0](n_nodes)
                nx.set_edge_attributes(g, 2, 'weight')

                cbs_th = 2*model[1](n_nodes)

                # shuffle labels
                random_labeling = generate_random_labeling(g)
                g = nx.relabel_nodes(g, random_labeling)

                assert cyclic_bandwidth_sum(g, weight_attr='weight') >= cbs_th

                # relabel graph according to cbs
                mach_labeling = get_mach_labeling(g, weight_attr='weight')
                g = nx.relabel_nodes(g, mach_labeling)

                assert cyclic_bandwidth_sum(g, weight_attr='weight') == cbs_th

    def test_weighted_graph(self):

        m = 4
        n = 3
        g = nx.cartesian_product(nx.cycle_graph(m), nx.cycle_graph(n))

        nx.set_edge_attributes(g, 1, 'weight')

        for i in range(m):
            for j in range(n):
                if i % 2 == 0:
                    if j < (n-1):
                        g[(i, j)][(i, j+1)]['weight'] = 100
                    else:
                        if i < (m-1):
                            g[(i, j)][(i+1, j)]['weight'] = 100
                        else:
                            g[(i, j)][(0, j)]['weight'] = 100
                else:
                    if j > 0:
                        g[(i, j)][(i, j-1)]['weight'] = 100
                    else:
                        if i < (m-1):
                            g[(i, j)][(i+1, j)]['weight'] = 100
                        else:
                            g[(i, j)][(0, j)]['weight'] = 100

        g = nx.convert_node_labels_to_integers(g)
        cbs_lb = n*m * 100
        cbs_ub = cbs_lb + (n*m)**2/2

        # shuffle labels
        random_labeling = generate_random_labeling(g)
        g = nx.relabel_nodes(g, random_labeling)

        assert cyclic_bandwidth_sum(g, weight_attr='weight') >= cbs_lb

        # relabel graph according to cbs
        mach_labeling = get_mach_labeling(g, weight_attr='weight')
        g = nx.relabel_nodes(g, mach_labeling)

        assert cyclic_bandwidth_sum(g, weight_attr='weight') >= cbs_lb
        assert cyclic_bandwidth_sum(g, weight_attr='weight') <= cbs_ub
