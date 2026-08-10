"""On-disk cache of traffic instances solved to equilibrium -- the expensive stage, and nothing else.

``download()`` draws instances and **solves** each to equilibrium (``solve_fn``, e.g. ~2.5 s per traffic
instance), storing the result under ``equilibrium`` -- a name that does not collide with the sampled
domain point ``model_input`` sets. It writes two raw files:

- ``base_graph.pt`` -- the canonical graph, unsolved, carrying ``game`` and the coupling matrices.
- ``instances.pt`` -- the dataset's solved instances. **The only ones a consumer ever splits.**

This class deliberately defines **no** ``process()``. PyG's ``has_process`` is
``overrides_method(cls, 'process')``, so it is ``False`` here and ``_process()`` never runs -- no ``processed/``
directory, no ``pre_transform.pt``, nothing written. That is the point: ``instances.pt`` already *is* the solved
instances with no operator examples attached, so a processed pass-through file would be a byte-identical second
copy of it. ``operator_datasets.OperatorDataset`` extends ``download()`` with a disjoint calibration set and
its subclasses add a ``process()`` (hence a processed file) each, one per point source, over the same
``instances.pt`` -- so both sources can be built from one set of solves and describe identical instances,
equilibria and calibration box.

Consumers pick the stage they need. The field model wants examples, so it constructs one of those subclasses.
Anything that wants only equilibria -- the solution model, the tuning scripts -- constructs *this* class, which
reads raw and writes nothing.

Two invariants worth stating, because both were once violated:

- ``equilibrium`` always means "this instance's own solved equilibrium". The base graph deliberately
  carries none: ``sample_params`` clones it, so a base-graph equilibrium would be inherited by every noised
  instance as though it were its own -- right shape, plausible magnitude, and silently wrong for the two
  things that read it (``calibration_kwargs`` and the ``rel_dist`` reference).
- ``base_graph.game`` records which VI family generated the root, so a reader derives its family from the
  data rather than being told. Being told is the same silent-mismatch hazard as the coupling matrices: a
  wrong ``--game`` would condition the model on the wrong family with nothing to complain.
"""

import torch
import torch_geometric.data
import tqdm


class EquilibriumDataset(torch_geometric.data.InMemoryDataset):
    """Noised traffic instances solved to equilibrium. Raw only -- see the module docstring.

    Args:
        root: directory for the raw caches (``base_graph.pt``, ``instances.pt``).
        base_graph: canonical graph to noise instances from. Cached as-is: unsolved, but carrying the
            asymmetric coupling matrices and ``game``.
        sample_fn: zero-arg callable returning a fresh noised instance (e.g. ``family.sample_params``).
        solve_fn: callable ``instance -> equilibrium`` tensor (e.g. ``family.solve_instance``).
        n_instances: how many instances to draw and solve into the dataset.
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
        quiet=False,
        **kwargs,
    ):
        self.base_graph = base_graph
        self.sample_fn = sample_fn
        self.solve_fn = solve_fn
        self.n_instances = n_instances
        self.quiet = quiet

        # super().__init__ runs download() (and, in a subclass, process()) if their caches are missing;
        # after it returns the raw files are guaranteed to exist.
        super().__init__(root, **kwargs)
        self.base_graph = self.raw_base_graph()
        self.load_instances()

    @property
    def raw_file_names(self):
        return ["base_graph.pt", "instances.pt"]

    @property
    def processed_file_names(self):
        # No processed artifact: this class defines no process(), so PyG never creates one. Declared as an
        # empty list purely so processed_paths is [] rather than the inherited NotImplementedError.
        return []

    def download(self):
        """Draw and solve the instances -- the expensive stage, cached so nothing below re-pays it."""
        assert self.base_graph is not None and self.sample_fn is not None and self.solve_fn is not None, (
            f"No cache at {self.raw_paths}. Pass base_graph, sample_fn, solve_fn and n_instances to build it."
        )
        assert self.n_instances is not None, "n_instances is required to solve a fresh dataset"
        torch.save(self.base_graph, self.raw_paths[0])
        torch.save(self._solve(self.n_instances, "solving"), self.raw_paths[1])

    def _solve(self, n, desc):
        """``n`` freshly drawn instances, each solved to equilibrium in place."""
        progress = range(n) if self.quiet else tqdm.trange(n, desc=desc)
        instances = []
        for _ in progress:
            instance = self.sample_fn()
            instance.equilibrium = self.solve_fn(instance)
            instances.append(instance)
        return instances

    def raw_base_graph(self):
        """The cached canonical graph. Read from raw rather than taken from the constructor argument because
        ``process()`` runs *inside* ``super().__init__()``, where that argument is still ``None`` on a read."""
        return torch.load(self.raw_paths[0], weights_only=False)

    def solved_instances(self):
        """The dataset's solved instances, straight from raw."""
        return torch.load(self.raw_paths[1], weights_only=False)

    def load_instances(self):
        """Populate the in-memory store from raw: this class attaches nothing, so there is no processed
        artifact to read (and no ``process()``, so PyG never creates one). Subclasses override this with
        ``self.load(self.processed_paths[0])``.

        Not optional. With ``has_process`` false, ``InMemoryDataset.__init__`` leaves ``slices`` as ``None``,
        and in that state ``len()`` returns **1** and ``get(0)`` returns ``copy.copy(None)`` -- silently
        broken rather than an error.
        """
        # collate + assigning data/slices is exactly what InMemoryDataset.load does, minus the file read;
        # both are public API (collate is a staticmethod, and load itself goes through the data setter).
        self.data, self.slices = self.collate(self.solved_instances())
