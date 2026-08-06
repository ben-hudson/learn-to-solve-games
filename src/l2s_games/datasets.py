"""On-disk cache of traffic instances solved to equilibrium, optionally with operator examples attached.

Two stages, mapped onto PyG's raw/processed split because their costs differ by ~1800x:

- ``download()`` draws ``n_instances`` noised instances and **solves** each to user equilibrium
  (``solve_fn``, ~2.5 s per instance), storing the result under ``equilibrium_cost`` /
  ``equilibrium_flow`` -- names chosen so they do not collide with the sampled domain point ``.cost`` that
  ``model_input`` sets. This is the expensive artifact, and it is genuinely *raw*: everything downstream is
  derived from it.
- ``process()`` optionally attaches **operator examples** to each instance -- the cost points a field model
  trains on plus the operator's value there (~1.4 ms per point). Injected as ``evaluate_fn``; omit it and
  the instances pass through untouched.

That staging is the point rather than an accident of the API. ``_download`` skips whenever the raw files
exist and *ignores* ``force_reload``, while ``_process`` honours it -- so changing how many points to draw,
or switching from uniformly-sampled points to expert-rollout ones, regenerates the examples and **reuses the
solves**. Had both lived in ``process()``, every such change would re-pay the solving.

It also splits the arguments by stage, which is how to read the constructor: ``sample_fn`` / ``solve_fn`` /
``n_instances`` build the raw artifact; ``evaluate_fn`` derives the processed one. A later read --
``EquilibriumDataset(root)`` with no callables -- just reloads.

Consumers pick the stage they need. The field model wants examples, so it wraps the processed instances in
``operator_datasets.OperatorExamples``. The solution model wants only equilibria, so it reads
``raw/instances.pt`` and never loads the (much larger) example tensors at all.

Two invariants worth stating, because both were once violated:

- ``equilibrium_cost`` always means "this instance's own solved equilibrium". The base graph deliberately
  carries none: ``sample_params`` clones it, so a base-graph equilibrium would be inherited by every noised
  instance as though it were its own -- right shape, plausible magnitude, and silently wrong for the two
  things that read it (``calibrate_ceiling`` and the ``rel_dist`` reference).
- ``base_graph.game`` records which VI family generated the root, so a reader derives its family from the
  data rather than being told. Being told is the same silent-mismatch hazard as the coupling matrices: a
  wrong ``--game`` would condition the model on the wrong family with nothing to complain.
"""

import torch
import torch_geometric.data
import tqdm


class EquilibriumDataset(torch_geometric.data.InMemoryDataset):
    """Noised traffic instances solved to equilibrium, optionally carrying operator examples.

    Args:
        root: directory for the raw (``base_graph.pt``, ``instances.pt``) and processed
            (``operator_examples.pt``) caches.
        base_graph: canonical graph to noise instances from. Cached as-is: unsolved, but carrying the
            asymmetric coupling matrices and ``game`` (see the module docstring).
        sample_fn: zero-arg callable returning a fresh noised instance (e.g. ``family.sample_params``).
        solve_fn: callable ``instance -> (cost, flow)`` equilibrium solver (e.g. ``PUMESolver.solve``).
        n_instances: how many instances to draw and solve.
        evaluate_fn: optional ``instances -> [OperatorEvaluations]``, one per instance, called with the
            **solved** instances -- so it can calibrate its own sampling range from their equilibria (see
            ``operator_datasets.POINT_SOURCES``). Omit for a solve-only dataset.
        quiet: suppress the progress bars.
        **kwargs: forwarded to ``InMemoryDataset.__init__``.
    """

    def __init__(
        self,
        root,
        base_graph=None,
        sample_fn=None,
        solve_fn=None,
        n_instances=None,
        evaluate_fn=None,
        quiet=False,
        **kwargs,
    ):
        self.base_graph = base_graph
        self.sample_fn = sample_fn
        self.solve_fn = solve_fn
        self.n_instances = n_instances
        self.evaluate_fn = evaluate_fn
        self.quiet = quiet

        # super().__init__ runs download()/process() if the respective caches are missing; after it
        # returns both are guaranteed to exist, so we can load them.
        super().__init__(root, **kwargs)
        self.load(self.processed_paths[0])
        self.base_graph = torch.load(self.raw_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["base_graph.pt", "instances.pt"]

    @property
    def processed_file_names(self):
        return ["operator_examples.pt"]

    def download(self):
        """Draw and solve the instances -- the expensive stage, cached so nothing below re-pays it."""
        assert self.base_graph is not None and self.sample_fn is not None and self.solve_fn is not None, (
            f"No cache at {self.raw_paths}. Pass base_graph, sample_fn, solve_fn and n_instances to build it."
        )
        assert self.n_instances is not None, "n_instances is required to solve a fresh dataset"
        torch.save(self.base_graph, self.raw_paths[0])
        progress = range(self.n_instances) if self.quiet else tqdm.trange(self.n_instances, desc="solving")
        instances = []
        for _ in progress:
            instance = self.sample_fn()
            instance.equilibrium_cost, instance.equilibrium_flow = self.solve_fn(instance)
            instances.append(instance)
        torch.save(instances, self.raw_paths[1])

    def process(self):
        """Attach each instance's operator examples.

        ``evaluate_fn`` receives every solved instance at once rather than one at a time, because it needs
        the equilibria as a population: the sampling range it draws points over is calibrated from them.
        """
        instances = torch.load(self.raw_paths[1], weights_only=False)
        if self.evaluate_fn is not None:
            for instance, evaluations in zip(instances, self.evaluate_fn(instances)):
                instance.points = evaluations.points
                instance.targets = evaluations.targets
                instance.preconditioner_diagonal = evaluations.preconditioner_diagonal
        self.save(instances, self.processed_paths[0])
