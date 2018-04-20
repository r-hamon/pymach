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
"""Utils functions.


References
----------
.. [Ham2016] Hamon, R., Borgnat, P., Flandrin, P., & Robardet, C. (2016).
   Relabelling vertices according to the network structure by minimizing the
   cyclic bandwidth sum. Journal of Complex Networks, 4(4), 534-560.

.. moduleauthor:: Ronan Hamon
"""
import numpy as np


def cbs_path(n):
    return n - 1


def cbs_cycle(n):
    return n


def cbs_wheel(n):
    return n + np.floor(1/4 * n**2)


def cbs_power_graph_cycles(n, k):
    return 0.5 * n * k * (k+1)


def cbs_complete_bipartite_graph(m, n):
    """

    For a complete bipartite graph $K_{n_1n_2}$, the optimal value of CBS is
    given by:
    .. math:
        \begin{array}{l l}
            \frac{n_1n_2^2 + n_1^2n_2}{4} & \quad \text{if $n_1$ and $n_2$ are
            even}\\
            \frac{n_1n_2^2 + n_1^2n_2 + n_1}{4} & \quad \text{if $n_1$ is even
            and $n_2$ is odd}\\
            \frac{n_1n_2^2 + n_1^2n_2 + n_1 + n_2}{4} & \quad \text{if $n_1$
            and $n_2$ are odd}\\
            \frac{n_1n_2^2 + n_1^2n_2 + n_2}{4} & \quad \text{if $n_1$ is odd
            and $n_2$ is even}\\
        \end{array} \right.
        $$
    """
    mn2 = m*(n**2)
    m2n = (m**2) * n

    return ((m % 2 == 0 and n % 2 == 0)*(mn2 + m2n)/4 + (m % 2 == 0 and n % 2
                                                         == 1)*(mn2 + m2n +
                                                                m)/4 + (m % 2
                                                                        == 1
                                                                        and n %
                                                                        2 ==
                                                                        0)*(mn2
                                                                            +
                                                                            m2n
                                                                            +
                                                                            n)/4
            + (m % 2 == 1 and n % 2 == 1)*(mn2 + m2n + m + n)/4)
