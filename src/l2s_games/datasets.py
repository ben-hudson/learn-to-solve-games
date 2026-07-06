"""On-disk cache of solved traffic instances.

Solving a noised instance to user equilibrium (``PUMESolver``) is expensive, so we pay it once,
offline, and persist the results as a PyG ``InMemoryDataset``. ``download`` solves the base graph
(cached in ``base_graph.pt``); ``process`` draws ``n_instances`` noised instances via ``sample_fn``,
solves each with ``solve_fn`` (storing the equilibrium under ``equilibrium_cost`` / ``equilibrium_flow``
so it does not collide with the sampled domain point ``.cost`` set by ``model_input``), and caches the
list in ``instances.pt``. A later read -- ``SolvedInstanceDataset(root)`` with no callables -- just
reloads the cache. The training script splits the loaded instances into bootstrap / val / test and
calibrates the streaming sampling range from the bootstrap equilibria (see ``calibrate_range``).
"""

import torch
import torch_geometric.data
import tqdm


class SolvedInstanceDataset(torch_geometric.data.InMemoryDataset):
    """In-memory cache of noised traffic instances, each solved to equilibrium.

    Args:
        root: directory for the raw (``base_graph.pt``) and processed (``instances.pt``) caches.
        base_graph: canonical base graph to solve and noise (required to build the cache).
        sample_fn: zero-arg callable returning a fresh noised instance (e.g. ``family.sample_params``).
        solve_fn: callable ``instance -> (cost, flow)`` equilibrium solver (e.g. ``PUMESolver.solve``).
        n_instances: number of instances to generate.
        quiet: suppress the generation progress bar.
        **kwargs: forwarded to ``InMemoryDataset.__init__``.
    """

    def __init__(
        self,
        root,
        base_graph=None,
        sample_fn=None,
        solve_fn=None,
        n_instances=None,
        quiet=False,
        **kwargs,
    ):
        self.base_graph = base_graph
        self.sample_fn = sample_fn
        self.solve_fn = solve_fn
        self.n_instances = n_instances
        self.quiet = quiet

        # super().__init__ runs download()/process() if the caches are missing; after it returns the
        # files are guaranteed to exist, so we can load them.
        super().__init__(root, **kwargs)
        self.load(self.processed_paths[0])
        self.base_graph = torch.load(self.raw_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["base_graph.pt"]

    @property
    def processed_file_names(self):
        return ["instances.pt"]

    def download(self):
        """Solve the base graph and save it as the raw artifact."""
        assert self.base_graph is not None and self.solve_fn is not None, (
            f"No cache at {self.raw_paths}. Pass base_graph and solve_fn to build the dataset."
        )
        self.base_graph.equilibrium_cost, self.base_graph.equilibrium_flow = self.solve_fn(self.base_graph)
        torch.save(self.base_graph, self.raw_paths[0])

    def process(self):
        """Generate noised instances, solve each to equilibrium, and cache them."""
        assert self.sample_fn is not None and self.solve_fn is not None and self.n_instances is not None, (
            f"No cache at {self.processed_paths}. Pass sample_fn, solve_fn and n_instances to build the dataset."
        )
        progress = range(self.n_instances) if self.quiet else tqdm.trange(self.n_instances)
        instances = []
        for _ in progress:
            instance = self.sample_fn()
            instance.equilibrium_cost, instance.equilibrium_flow = self.solve_fn(instance)
            instances.append(instance)
        self.save(instances, self.processed_paths[0])
