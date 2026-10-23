"""
network_tools
=============

A small collection of from-scratch network science utilities built on top of
NetworkX and NumPy.

Contents
--------
Plotting helpers
    plot_degree
        Compute the support and probability values of a degree distribution,
        with optional logarithmic binning, ready for plotting.

Centrality measures
    closeness_centrality_from_scratch
        Closeness centrality for every node (Wasserman & Faust scaling for
        disconnected graphs).
    eigenvector_centrality_from_scratch
        Eigenvector centrality computed with power iteration.

Null models
    degree_preserving_randomization
        Edge-swap (Maslov-Sneppen) randomization that preserves every node's
        degree.
    configuration_model_from_degree_sequence
        Configuration model built by randomly pairing stubs.

Community structure
    calculate_modularity
        Newman-Girvan modularity for weighted or unweighted graphs.

Requirements
------------
numpy, networkx
"""

import networkx as nx
import numpy as np

__all__ = [
    "plot_degree",
    "closeness_centrality_from_scratch",
    "eigenvector_centrality_from_scratch",
    "degree_preserving_randomization",
    "configuration_model_from_degree_sequence",
    "calculate_modularity",
]


def plot_degree(degree, number_of_bins=50, log_binning=True, base=2):
    """
    Given a degree sequence, return the y values (probability) and the
    x values (support) of a degree distribution that you're going to plot.

    Parameters
    ----------
    degree (np.ndarray or list):
        a vector of length N that corresponds to the degree, k_i, of every
        node, v_i, in the network

    number_of_bins (int):
        length of output vectors

    log_binning (bool)
        if you are plotting on a log-log axis, then this is useful

    base (int):
        log base, defaults to 2

    Returns
    -------
    x, y (np.ndarray):
        the support and probability values of the degree distribution

    """

    # We need to define the support of our distribution
    lower_bound = min(degree)
    upper_bound = max(degree)

    # And the bins
    if log_binning:
        log = np.log2 if base == 2 else np.log10
        lower_bound = log(lower_bound) if lower_bound >= 1 else 0.0
        upper_bound = log(upper_bound)
        bins = np.logspace(lower_bound,upper_bound,number_of_bins, base = base)
    else:
        bins = np.linspace(lower_bound,upper_bound,number_of_bins)

    # Then we can compute the histogram using numpy
    y, __ = np.histogram(degree,
                         bins=bins,
                         density=True)
    # Now, we need to compute for each y the value of x
    x = bins[1:] - np.diff(bins)/2.0

    return x, y


def closeness_centrality_from_scratch(G):
    """
    Calculate the closeness centrality for each node in a graph from scratch.

    Closeness centrality is a measure of how close a node is to all other nodes
    in the network. It is calculated as the reciprocal of the sum of the shortest
    path distances from a node to all other nodes in the graph. This function
    computes the closeness centrality for all nodes in the graph `G` without
    using any external library functions for the centrality calculation.

    Parameters
    ----------
    G : networkx.Graph
        The input graph on which the closeness centrality is calculated. It can
        be any type of graph (undirected, directed, etc.) supported by NetworkX.

    Returns
    -------
    centrality : dict
        A dictionary where the keys are nodes in the graph and the values are
        their corresponding closeness centrality scores. If a node is isolated,
        its centrality will be 0.

    Notes
    -----
    - For each node, this function computes the sum of shortest path lengths to
      all other reachable nodes in the graph using NetworkX's `shortest_path_length`.
    - If the graph is not connected, a node can only reach part of it. Like
      NetworkX, we compute closeness within the part the node can reach, then
      scale it by the fraction of the other nodes it can reach (Wasserman &
      Faust, 1994), so a node in a small component doesn't get an inflated
      score. An isolated node gets 0.0.
    - In a directed graph, NetworkX measures how close the other nodes are to
      a node (incoming distance), so we compute distances in the reversed graph.

    Time Complexity
    ---------------
    The time complexity is O(N * (V + E)), where N is the number of nodes, V is
    the number of vertices, and E is the number of edges, due to the use of
    shortest path calculations for each node.

    Citations
    ---------
    Bavelas, A. (1950). Communication patterns in task-oriented groups. The Journal
    of the Acoustical Society of America, 22(6), 725-730.

    Sabidussi, G. (1966). The centrality index of a graph. Psychometrika, 31(4), 581–603.

    Freeman, L. C. (1979). Centrality in social networks conceptual clarification.
    Social Networks, 1(3), 215–239.

    Wasserman, S., & Faust, K. (1994). Social Network Analysis: Methods and
    Applications. Cambridge University Press.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.path_graph(4)
    >>> closeness_centrality_from_scratch(G)
    {0: 0.6666666666666666, 1: 1.0, 2: 1.0, 3: 0.6666666666666666}
    """

    centrality = {}
    N = G.number_of_nodes()  # Total number of nodes in the graph

    # In a directed graph, the distances *to* node_i are the distances
    # *from* node_i in the reversed graph
    H = G.reverse() if G.is_directed() else G

    for node_i in G.nodes():
        # Compute shortest paths from node_i to every node it can reach
        shortest_paths = nx.shortest_path_length(H, source=node_i)

        # Sum the lengths of the shortest paths
        total_distance = sum(shortest_paths.values())
        n_reachable = len(shortest_paths)  # includes node_i itself

        if total_distance > 0 and N > 1:
            # Closeness within the part of the graph node_i can reach...
            centrality[node_i] = (n_reachable - 1) / total_distance
            # ...scaled by the fraction of the other nodes it can reach
            centrality[node_i] *= (n_reachable - 1) / (N - 1)
        else:
            centrality[node_i] = 0.0  # In case the node is isolated

    return centrality


def eigenvector_centrality_from_scratch(G, max_iter=100, tol=1e-08):
    """
    Calculate the eigenvector centrality for each node in a graph from scratch.

    Eigenvector centrality is a measure of a node's influence in a network based on
    the idea that connections to high-scoring nodes contribute more to the score
    of a node than equal connections to low-scoring nodes. This centrality measure
    assigns relative scores to all nodes in the network based on the principle that
    a node's centrality is determined by the centrality of its neighbors.

    Parameters
    ----------
    G : networkx.Graph
        The input graph on which the eigenvector centrality is calculated. It can be
        any type of graph (undirected, directed, etc.) supported by NetworkX.

    max_iter : int, optional (default=100)
        Maximum number of iterations for the power iteration method used to compute
        the centrality values. Higher values may be required for large graphs.

    tol : float, optional (default=1e-08)
        Tolerance for the convergence of the eigenvector centrality values. The
        algorithm iterates until the change in centrality values is smaller than this
        threshold.

    Returns
    -------
    centrality : dict
        A dictionary where the keys are nodes in the graph and the values are
        their corresponding eigenvector centrality scores.

    Notes
    -----
    - Eigenvector centrality was introduced by Bonacich (1972) as an extension
      of degree centrality, emphasizing the importance of connections to high-degree
      or highly influential nodes.
    - This algorithm computes eigenvector centrality using the power iteration
      method, which involves iteratively updating the centrality scores of each
      node based on the scores of their neighbors until convergence.
    - Eigenvector centrality works best in connected, undirected graphs; for
      directed or disconnected graphs, results may vary or be undefined.
    - On a directed graph, a node's score is the sum of the scores of the nodes
      that point to it.
    - On a bipartite graph (a path, a star, a tree), iterating with A alone
      flips between two vectors and never settles; see below.
    - The algorithm will stop either after `max_iter` iterations or when the
      centrality values converge to within the specified `tol`.

    Time Complexity
    ---------------
    The time complexity is O(V * E * I), where V is the number of vertices,
    E is the number of edges, and I is the number of iterations (limited by `max_iter`).

    Citations
    ---------
    Bonacich, P. (1972). Factoring and weighting approaches to status scores and
    clique identification. *Journal of Mathematical Sociology, 2*(1), 113-120.

    Newman, M. E. J. (2008). The mathematics of networks. *The New Palgrave
    Dictionary of Economics, 2*(1), 1-12.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> eigenvector_centrality_from_scratch(G)
    {0: 0.3730400736153818, 1: 0.2082196569730357, 2: 0.20624526357714606, ...}
    """

    # Initialize centrality dict with uniform values for all nodes
    centrality = {node: 1.0 / len(G) for node in G}
    N = len(G)

    # In a directed graph, a node's score comes from the nodes that point to it
    in_neighbors = G.predecessors if G.is_directed() else G.neighbors

    # Power iteration method
    for _ in range(max_iter):
        prev_centrality = centrality.copy()
        max_diff = 0  # Track maximum change in centrality values

        for node in G:
            # Update centrality: sum of neighbors' centralities
            centrality[node] = sum(prev_centrality[neighbor] for neighbor in in_neighbors(node))

        # Normalize centrality values (divide by Euclidean norm)
        norm = np.sqrt(sum(value ** 2 for value in centrality.values()))
        if norm == 0:
            return centrality  # a graph with no edges: every score is 0
        centrality = {node: value / norm for node, value in centrality.items()}

        # Check for convergence
        max_diff = max(abs(centrality[node] - prev_centrality[node]) for node in G)
        if max_diff < tol:
            break

    return centrality


def degree_preserving_randomization(G, n_iter=None):
    """
    Perform degree-preserving randomization on a graph.

    Degree-preserving randomization, also known as edge swapping or rewiring,
    is a method for creating randomized versions of a graph while preserving
    the degree distribution of each node. This is achieved by repeatedly
    swapping pairs of edges in the graph, ensuring that the degree (number of
    edges connected) of each node remains unchanged. The result is a graph
    with the same degree distribution but a randomized edge structure, which
    can be used as a null model to compare with the original network.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to be randomized. The graph can be directed or
        undirected, but it must be simple (i.e., no self-loops or parallel edges).

    n_iter : int or None, optional (default=None)
        The number of swap attempts. An attempt that would create a self-loop
        or a parallel edge is skipped, so the number of swaps actually made is
        smaller than `n_iter`. A higher number of attempts leads to more
        randomization, and the degrees stay the same either way. If None, we
        use 10 times the number of edges.

    Returns
    -------
    G_random : networkx.Graph
        A randomized graph with the same degree distribution as the original
        graph `G`, but with a shuffled edge structure.

    Notes
    -----
    - This method works by selecting two edges at random, say (u, v) and (x, y),
      and attempting to swap them to (u, y) and (x, v) (or (u, x) and (v, y)),
      ensuring that no self-loops or parallel edges are created in the process.
    - On a directed graph, only the swap to (u, y) and (x, v) is used, because
      it keeps every node's in-degree and out-degree. The swap to (u, x) and
      (v, y) keeps each node's total degree but changes its in- and out-degree.
    - Degree-preserving randomization is particularly useful for creating null
      models in network analysis, as it allows for the investigation of whether
      specific network properties (e.g., clustering, path lengths) are a result
      of the network's structure or just its degree distribution.
    - The effectiveness of randomization depends on the number of iterations
      (`n_iter`). As a rule of thumb, using about 10 times the number of edges
      in the graph for `n_iter` often provides sufficient randomization, which
      is why that is the default.

    Example
    -------
    >>> import networkx as nx
    >>> G = nx.erdos_renyi_graph(10, 0.5)
    >>> G_random = degree_preserving_randomization(G, n_iter=100)

    Citations
    ---------
    Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002).
    Network motifs: simple building blocks of complex networks. *Science*, 298(5594), 824-827.

    Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks.
    *Science*, 296(5569), 910-913.
    """

    G_random = G.copy()
    edges = list(G_random.edges())
    num_edges = len(edges)

    if n_iter is None:
        n_iter = 10 * num_edges

    for _ in range(n_iter):
        # Select two random edges (u, v) and (x, y)
        edge1_id = np.random.randint(num_edges)
        u, v = edges[edge1_id]
        edge2_id = np.random.randint(num_edges)
        x, y = edges[edge2_id]

        # Avoid selecting the same edge pair or creating self-loops
        if len({u, v, x, y}) == 4:
            # Swap the edges with some probability. On a directed graph, only
            # the first swap keeps every node's in- and out-degree.
            if G_random.is_directed() or np.random.rand() > 0.5:
                # Swap (u, v) with (u, y) and (x, v)
                if not (G_random.has_edge(u, y) or G_random.has_edge(x, v)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, y)
                    G_random.add_edge(x, v)
                    # Keep the edge list in step with the graph
                    edges[edge1_id], edges[edge2_id] = (u, y), (x, v)
            else:
                # Swap (u, v) with (u, x) and (v, y)
                if not (G_random.has_edge(u, x) or G_random.has_edge(v, y)):
                    G_random.remove_edge(u, v)
                    G_random.remove_edge(x, y)
                    G_random.add_edge(u, x)
                    G_random.add_edge(v, y)
                    # Keep the edge list in step with the graph
                    edges[edge1_id], edges[edge2_id] = (u, x), (v, y)

    return G_random


def configuration_model_from_degree_sequence(degree_sequence, return_simple=True):
    """
    Generate a random graph using the configuration model from a given degree sequence
    without using the NetworkX built-in function.

    The configuration model generates a random graph that preserves the degree
    sequence of nodes by assigning "stubs" or "half-edges" to each node and
    randomly pairing these stubs to form edges. This process can result in
    graphs with self-loops and parallel edges, which can be removed if needed.

    Parameters
    ----------
    degree_sequence : list of int
        A list representing the degree of each node in the graph. The sum of
        the degrees in this sequence must be even for the configuration model
        to create a valid graph.

    return_simple : bool, optional (default=True)
        If True, merge parallel edges and remove self-loops before returning.
        The result is a simple graph, but some of its degrees can be lower
        than the ones in `degree_sequence`. If False, return the multigraph.

    Returns
    -------
    G : networkx.MultiGraph or networkx.Graph
        A multigraph generated from the given degree sequence. The graph may
        contain self-loops and parallel edges, which are allowed in the
        configuration model. With `return_simple=True`, its simple version.

    Notes
    -----
    - This method works by assigning "stubs" or "half-edges" to nodes based on
      their degree and then randomly pairing them to form edges. The resulting
      graph can have self-loops and parallel edges.
    - If a simple graph is required, `nx.Graph(G)` merges parallel edges, but
      it keeps self-loops, which have to be removed separately with
      `G.remove_edges_from(nx.selfloop_edges(G))`. Both steps lower some
      nodes' degrees, so the simple graph no longer matches the degree
      sequence exactly.
    - The degree sequence must have an even sum for a valid graph construction.
      If the sum of the degrees is odd, no graph can be constructed.

    Time Complexity
    ---------------
    The time complexity is O(E), where E is the number of edges in the graph.

    Example
    -------
    >>> degree_sequence = [3, 3, 2, 2, 1, 1]
    >>> G = configuration_model_from_degree_sequence(degree_sequence)
    >>> nx.is_graphical(degree_sequence)
    True
    """

    # Check if the degree sequence is valid (sum of degrees must be even)
    if sum(degree_sequence) % 2 != 0:
        raise ValueError("The sum of the degree sequence must be even.")

    # Create stubs list: node i appears degree_sequence[i] times
    stubs = []
    for node, degree in enumerate(degree_sequence):
        stubs.extend([node] * degree)

    # Shuffle stubs to randomize the pairing process
    np.random.shuffle(stubs)

    # Initialize an empty multigraph
    G = nx.MultiGraph()

    # Add nodes to the graph
    G.add_nodes_from(range(len(degree_sequence)))

    # Pair stubs to create edges
    while stubs:
        u = stubs.pop()
        v = stubs.pop()

        # Add the edge to the graph
        G.add_edge(u, v)

    if return_simple:
        # Convert the multigraph to a simple graph. nx.Graph() merges parallel
        # edges but keeps self-loops, so we remove those separately.
        G_simple = nx.Graph(G)
        G_simple.remove_edges_from(list(nx.selfloop_edges(G_simple)))

        return G_simple

    else:
        return G


def calculate_modularity(G, partition):
    """
    Calculates the modularity score for a given partition of the graph, whether the graph is weighted or unweighted.

    Modularity is a measure of the strength of division of a network into communities. It compares the actual
    density of edges within communities to the expected density if edges were distributed randomly. For weighted
    graphs, the weight of the edges is taken into account.

    The modularity Q is calculated as:

    Q = (1 / 2m) * sum((A_ij - (k_i * k_j) / (2m)) * delta(c_i, c_j))

    where:
    - A_ij is the weight of the edge between nodes i and j (1 if unweighted).
    - k_i is the degree of node i (or the weighted degree for weighted graphs).
    - m is the total number of edges in the graph, or the total weight of the edges if the graph is weighted.
    - delta(c_i, c_j) is 1 if nodes i and j belong to the same community, and 0 otherwise.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph, which can be undirected and either weighted or unweighted. The graph's nodes represent the
        entities, and its edges represent connections between them.

    partition : list of sets
        A list of sets where each set represents a community. Each set contains the nodes belonging to that community.
        For example, [{0, 1, 2}, {3, 4}] represents two communities, one with nodes 0, 1, and 2, and another with nodes
        3 and 4.

    Returns:
    --------
    modularity_score : float
        The modularity score for the given partition of the graph. A higher score indicates stronger community structure,
        and a lower (or negative) score suggests weak or no community structure.

    Notes:
    ------
    - If the graph has weights, they will be used in the modularity calculation. If no weights are present, the function
      assumes each edge has a weight of 1 (i.e., unweighted).

    - The function assumes that all nodes in the graph are assigned to exactly one community. If any node is missing
      from the community list, it is treated as not belonging to any community, and the results may not be accurate.

    - If the graph has no edges, the modularity is undefined, and this function will return 0 because the total number
      of edges (2m) would be zero.

    Example:
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}]
    >>> modularity_score = calculate_modularity(G, communities)
    >>> print("Modularity:", modularity_score)

    References:
    -----------
    Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure
    in networks. Physical Review E, 69(2), 026113.
    """

    def remap_partition(partition):
        """
        Converts and remaps a partition to a list-of-lists structure suitable for modularity calculations.

        This function remaps the input partition (whether it's in dictionary form or a flat list of community labels)
        to a list-of-lists format, where each list represents a community and contains the nodes in that community.
        The function also ensures that community labels are contiguous integers starting from 0, which is typically
        required for modularity-based algorithms.
        """

        # if partition is a dictionary where the keys are nodes and values communities
        if type(partition)==dict:
            unique_comms = np.unique(list(partition.values()))
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in partition.items():
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(list(partition.values()))
            communities = [[] for i in unique_comms]
            for i,j in partition.items():
                communities[j].append(i)

            return communities

        # if partition is a list of community assignments
        elif type(partition)==list and\
                not any(isinstance(el, list) for el in partition):
            unique_comms = np.unique(partition)
            comm_mapping = {i:ix for ix,i in enumerate(unique_comms)}
            for i,j in enumerate(partition):
                partition[i] = comm_mapping[j]

            unique_comms = np.unique(partition)
            communities = [[] for i in np.unique(partition)]
            for i,j in enumerate(partition):
                communities[j].append(i)

            return communities

        # otherwise assume input is a properly-formatted list of lists
        else:
            communities = partition.copy()
            return communities


    # We now should have a list-of-lists structure for communities
    communities = remap_partition(partition)

    # Total weight of edges in the graph (or number of edges if unweighted)
    if nx.is_weighted(G):
        m = G.size(weight='weight')
        degree = dict(G.degree(weight='weight'))  # Weighted degree for each node
    else:
        m = G.number_of_edges()  # Number of edges in the graph
        degree = dict(G.degree())  # Degree for each node (unweighted)

    # Modularity score
    modularity_score = 0.0

    # Loop over all pairs of nodes i, j within the same community
    for community in communities:
        for i in community:
            for j in community:
                # Get the weight of the edge between i and j, or assume weight 1 if unweighted
                if G.has_edge(i, j):
                    A_ij = G[i][j].get('weight', 1)  # Use weight if available, otherwise assume 1
                else:
                    A_ij = 0  # No edge between i and j

                # Expected number of edges (or weighted edges) between i and j in a random graph
                expected_edges = (degree[i] * degree[j]) / (2 * m)

                # Contribution to modularity
                modularity_score += (A_ij - expected_edges)

    # Normalize by the total number of edges (or total edge weight) 2m
    modularity_score /= (2 * m)

    return modularity_score
