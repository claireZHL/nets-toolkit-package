# Welcome to nets_toolkit_package

|        |        |
|--------|--------|
| Package | [![Latest PyPI Version](https://img.shields.io/pypi/v/nets_toolkit_package.svg)](https://pypi.org/project/nets_toolkit_package/) [![Supported Python Versions](https://img.shields.io/pypi/pyversions/nets_toolkit_package.svg)](https://pypi.org/project/nets_toolkit_package/)  |
| Meta   | [![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md) |

nets_toolkit_package is a small collection of from-scratch network science utilities built on [NetworkX](https://networkx.org/) and [NumPy](https://numpy.org/). It re-implements several standard tools directly from their definitions: degree-distribution binning, closeness and eigenvector centrality, two null models (degree-preserving randomization and the configuration model), and modularity.

The code is written to be readable and to follow the original papers, which makes it useful for teaching and for checking your understanding. For large graphs, the optimized NetworkX equivalents will be considerably faster.

## Contents

- [Get started](#get-started)
- [Function reference](#function-reference)
- [Summary table](#summary-table)
- [Known caveats](#known-caveats)
- [References](#references)
- [Copyright](#copyright)

## Get started

You can install this package into your preferred Python environment using pip:

```bash
$ pip install nets_toolkit_package
```

Requirements: Python 3, `numpy`, `networkx` (installed automatically with pip).

To work from a local copy instead, run this from the project folder:

```bash
$ pip install -e .
```

To use nets_toolkit_package in your code:

```python
import networkx as nx
import nets_toolkit_package as nt

G = nx.karate_club_graph()

# Centrality
closeness = nt.closeness_centrality_from_scratch(G)
eigenvector = nt.eigenvector_centrality_from_scratch(G)

# Null model: same degrees, shuffled edges
G_null = nt.degree_preserving_randomization(G)

# Community structure
communities = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}]
Q = nt.calculate_modularity(G, communities)
```

## Function reference

### `plot_degree`

```python
plot_degree(degree, number_of_bins=50, log_binning=True, base=2)
```

Computes the x and y values of a degree distribution so you can plot it yourself (for example with matplotlib).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `degree` | list or `np.ndarray` | required | Degree of every node in the network (length N). |
| `number_of_bins` | int | `50` | Number of bin edges (the output vectors have `number_of_bins - 1` entries). |
| `log_binning` | bool | `True` | Use logarithmically spaced bins. Useful for log-log plots of heavy-tailed distributions. |
| `base` | int | `2` | Log base used for the bin spacing. |

**Returns:** `x, y` (both `np.ndarray`), the support (bin centers) and the probability density at each bin.

**Notes**

- `y` comes from `np.histogram(..., density=True)`, so it is a probability *density* (normalized by bin width). With log bins this is what you want for a fair comparison across bins of different widths.
- `x` is the midpoint of each bin.
- If the minimum degree is below 1 (for example 0), the lower bound of the log bins is set to `base**0 = 1`.
- Only `base=2` and `base=10` are handled correctly. Any base other than 2 is treated as base 10 when computing the bounds.

**Example**

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.barabasi_albert_graph(5000, 3)
degrees = [d for _, d in G.degree()]
x, y = nt.plot_degree(degrees, number_of_bins=30, log_binning=True, base=10)

plt.loglog(x, y, "o")
plt.xlabel("degree k")
plt.ylabel("P(k)")
plt.show()
```

---

### `closeness_centrality_from_scratch`

```python
closeness_centrality_from_scratch(G)
```

Computes closeness centrality for every node, based on shortest-path distances. It uses `nx.shortest_path_length` for the distances but computes the centrality itself.

| Parameter | Type | Description |
|---|---|---|
| `G` | `networkx.Graph` (or directed graph) | The input graph. |

**Returns:** `dict` mapping each node to its closeness centrality. Isolated nodes get `0.0`.

**How it works**

For a node *i* that can reach `n_reachable` nodes (including itself) with total distance `d_total`:

```
closeness(i) = (n_reachable - 1) / d_total  *  (n_reachable - 1) / (N - 1)
```

- The first factor is the classic closeness within the part of the graph node *i* can reach.
- The second factor scales it by the fraction of other nodes it can reach (Wasserman & Faust, 1994), so a node in a small component does not get an inflated score. This matches NetworkX's default behavior (`wf_improved=True`).
- On a **connected** graph the second factor is 1, giving the textbook definition.

**Directed graphs:** NetworkX measures how close the other nodes are *to* a node (incoming distance), so distances are computed on the reversed graph.

**Complexity:** O(N · (V + E)), one shortest-path computation per node.

**Example**

```python
G = nx.path_graph(4)
nt.closeness_centrality_from_scratch(G)
# {0: 0.6666666666666666, 1: 1.0, 2: 1.0, 3: 0.6666666666666666}
```

---

### `eigenvector_centrality_from_scratch`

```python
eigenvector_centrality_from_scratch(G, max_iter=100, tol=1e-08)
```

Computes eigenvector centrality using power iteration. A node's score is proportional to the sum of its neighbors' scores, so connections to important nodes count for more.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G` | `networkx.Graph` (or directed graph) | required | The input graph. |
| `max_iter` | int | `100` | Maximum number of power-iteration steps. Larger graphs may need more. |
| `tol` | float | `1e-08` | Convergence threshold on the largest change in any node's score between iterations. |

**Returns:** `dict` mapping each node to its centrality, normalized to unit Euclidean norm. A graph with no edges returns all zeros.

**How it works**

1. Start every node at `1 / N`.
2. Replace each node's score with the sum of its neighbors' previous scores.
3. Normalize the vector by its Euclidean norm.
4. Stop when the largest change is below `tol`, or after `max_iter` iterations.

**Directed graphs:** a node's score is the sum of the scores of the nodes that *point to* it.

**Caveats**

- Works best on connected, undirected graphs. For directed or disconnected graphs, results may vary or be undefined.
- On **bipartite graphs** (paths, stars, trees), iterating with the adjacency matrix alone oscillates between two vectors and never converges. The function will simply stop at `max_iter` without a converged answer.
- If it does not converge within `max_iter`, the current estimate is returned without any warning. Check the result or increase `max_iter` if in doubt.

**Complexity:** O(V · E · I), where I is the number of iterations.

**Example**

```python
G = nx.karate_club_graph()
scores = nt.eigenvector_centrality_from_scratch(G)
```

---

### `degree_preserving_randomization`

```python
degree_preserving_randomization(G, n_iter=None)
```

Returns a randomized copy of a graph that keeps every node's degree. It works by repeatedly swapping the endpoints of pairs of edges (the Maslov-Sneppen edge-swap method), producing a null model with the same degree distribution but shuffled structure.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `G` | `networkx.Graph` or `DiGraph` | required | A **simple** graph (no self-loops or parallel edges). |
| `n_iter` | int or `None` | `None` | Number of swap *attempts*. If `None`, uses `10 * number_of_edges`. |

**Returns:** a new graph (`G_random`). The original `G` is not modified.

**How it works**

Pick two random edges `(u, v)` and `(x, y)`. If all four nodes are distinct, try one of these swaps (chosen at random for undirected graphs):

- `(u, v), (x, y)` → `(u, y), (x, v)`
- `(u, v), (x, y)` → `(u, x), (v, y)`

A swap is skipped if it would create a parallel edge. Because skipped attempts still count toward `n_iter`, the number of swaps actually performed is smaller than `n_iter`. Degrees are preserved regardless.

**Directed graphs:** only the first swap is used, since it preserves every node's in-degree and out-degree. The second swap preserves total degree but would change in- and out-degrees.

**Tips**

- The default of 10 × edges is a common rule of thumb for adequate mixing. Increase `n_iter` for more thorough randomization.
- The function uses NumPy's global random state, so call `np.random.seed(...)` beforehand if you need reproducible results.

**Example**

```python
G = nx.erdos_renyi_graph(10, 0.5)
G_random = nt.degree_preserving_randomization(G, n_iter=100)

assert dict(G.degree()) == dict(G_random.degree())
```

---

### `configuration_model_from_degree_sequence`

```python
configuration_model_from_degree_sequence(degree_sequence, return_simple=True)
```

Builds a random graph from a given degree sequence using the configuration model, without calling NetworkX's built-in generator. Each node gets as many "stubs" (half-edges) as its degree, and the stubs are paired uniformly at random.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `degree_sequence` | list of int | required | Desired degree for each node. The sum must be even. |
| `return_simple` | bool | `True` | If `True`, merge parallel edges and drop self-loops before returning. If `False`, return the raw multigraph. |

**Returns:** `networkx.Graph` if `return_simple=True`, otherwise `networkx.MultiGraph`. Nodes are labeled `0` to `len(degree_sequence) - 1`.

**Raises:** `ValueError` if the sum of the degree sequence is odd.

**Important:** the raw configuration model can produce self-loops and parallel edges. With `return_simple=True`, removing them lowers some nodes' degrees, so **the simple graph no longer matches the degree sequence exactly**. Use `return_simple=False` if you need the degrees preserved exactly (a self-loop counts twice toward a node's degree in a `MultiGraph`).

**Complexity:** O(E).

**Example**

```python
degree_sequence = [3, 3, 2, 2, 1, 1]

G_multi = nt.configuration_model_from_degree_sequence(degree_sequence, return_simple=False)
G_simple = nt.configuration_model_from_degree_sequence(degree_sequence)
```

---

### `calculate_modularity`

```python
calculate_modularity(G, partition)
```

Computes the Newman-Girvan modularity of a partition for weighted or unweighted undirected graphs:

```
Q = (1 / 2m) * sum_ij [ A_ij - (k_i * k_j) / (2m) ] * delta(c_i, c_j)
```

where `A_ij` is the edge weight between *i* and *j* (1 if unweighted), `k_i` is the (weighted) degree of node *i*, `m` is the total number of edges (or total edge weight), and `delta(c_i, c_j)` is 1 when *i* and *j* are in the same community.

| Parameter | Type | Description |
|---|---|---|
| `G` | `networkx.Graph` | Undirected graph, weighted or unweighted. Weights are read from the `'weight'` edge attribute. |
| `partition` | dict, list, or list of lists/sets | The community assignment (see formats below). |

**Accepted partition formats**

| Format | Example | Meaning |
|---|---|---|
| dict | `{0: "a", 1: "a", 2: "b"}` | node → community label |
| flat list | `["a", "a", "b"]` | community label of node *i* at position *i* |
| list of lists | `[[0, 1], [2]]` | one list of nodes per community |

An internal helper, `remap_partition`, converts the first two formats into a list of lists with contiguous community indices starting at 0.

**Returns:** `float`, the modularity score. Higher values indicate stronger community structure. Values near 0 or below suggest no community structure beyond chance.

**Caveats**

- Every node should belong to exactly one community. Missing nodes are ignored, which gives inaccurate results.
- A graph with no edges makes `2m` zero, which causes a division by zero.
- The helper relabels a dict or flat-list partition **in place**, so your original `partition` object is modified. Pass a copy if you need to keep the original labels.
- A flat list is interpreted as labels indexed by node position, so it only makes sense when nodes are labeled `0..N-1`.
- The doubly nested loop over community members is O(sum of community size²), which is fine for small and medium graphs but slow for large communities.

**Example**

```python
G = nx.karate_club_graph()
communities = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}]
Q = nt.calculate_modularity(G, communities)
print("Modularity:", Q)
```

## Summary table

| Function | Purpose | Input | Output |
|---|---|---|---|
| `plot_degree` | Bin a degree sequence for plotting | list/array of degrees | `x, y` arrays |
| `closeness_centrality_from_scratch` | Closeness centrality | graph | dict |
| `eigenvector_centrality_from_scratch` | Eigenvector centrality via power iteration | graph | dict |
| `degree_preserving_randomization` | Degree-preserving null model (edge swaps) | simple graph | graph |
| `configuration_model_from_degree_sequence` | Configuration-model null model | degree sequence | `Graph` or `MultiGraph` |
| `calculate_modularity` | Modularity of a partition | graph, partition | float |

## Known caveats

- These are educational implementations. For production work on large networks, prefer the NetworkX versions (`nx.closeness_centrality`, `nx.eigenvector_centrality`, `nx.double_edge_swap`, `nx.configuration_model`, `nx.community.modularity`).
- Randomized functions use NumPy's global random state. Set `np.random.seed(...)` for reproducibility.
- Eigenvector centrality may not converge on bipartite graphs and does not warn when it hits `max_iter`.
- `plot_degree` handles log bases 2 and 10 only.

## References

- Bavelas, A. (1950). Communication patterns in task-oriented groups. *The Journal of the Acoustical Society of America*, 22(6), 725-730.
- Sabidussi, G. (1966). The centrality index of a graph. *Psychometrika*, 31(4), 581-603.
- Freeman, L. C. (1979). Centrality in social networks conceptual clarification. *Social Networks*, 1(3), 215-239.
- Wasserman, S., & Faust, K. (1994). *Social Network Analysis: Methods and Applications*. Cambridge University Press.
- Bonacich, P. (1972). Factoring and weighting approaches to status scores and clique identification. *Journal of Mathematical Sociology*, 2(1), 113-120.
- Newman, M. E. J. (2008). The mathematics of networks. *The New Palgrave Dictionary of Economics*, 2(1), 1-12.
- Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002). Network motifs: simple building blocks of complex networks. *Science*, 298(5594), 824-827.
- Maslov, S., & Sneppen, K. (2002). Specificity and stability in topology of protein networks. *Science*, 296(5569), 910-913.
- Newman, M. E. J., & Girvan, M. (2004). Finding and evaluating community structure in networks. *Physical Review E*, 69(2), 026113.

## Copyright

- Copyright © 2026 Claire.
- Free software distributed under the [MIT License](./LICENSE).
