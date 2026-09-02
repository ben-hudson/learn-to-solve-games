"""Fully amortized SPE solver: prices straight off the instance data, trained self-supervised."""

import lightning as L
import torch

from l2s_games.envs.spe.game.cost_space import PriceSPE

from .readouts import PriceReadout


def ascent_field(instance: PriceSPE, prices: torch.Tensor):
    """``-Psi(u, v)``: the direction the prices move under the convergent dynamics.

    ``PriceSPE.operator`` is the monotone map in the standard variational-inequality sign, so every
    dynamic -- and every loss in ``l2s_games.losses`` -- wants its negation.
    """
    return -instance.operator(prices)


class SpeSolutionModel(L.LightningModule):
    """Predicts the equilibrium price vector directly from the instance data (full amortization).

    Self-supervised on the market-clearing field: an equilibrium is exactly ``Psi(u, v) = 0``, so no
    solver labels are needed and equilibrium non-uniqueness is a non-issue. Price space is what makes
    the whole objective a field surrogate. ``PriceSPE`` is an unconstrained equation, so there is no
    feasible set, no projection anywhere in the loop, and -- unlike the zero-sum case -- the
    surrogate's reported value really does fall to zero at a solution, which makes ``train/loss`` a
    fit signal rather than just a reported number.

    The surrogate's substituted gradient earns its keep twice over here. ``Psi``'s Jacobian is a
    ``1/delta``-weighted bipartite Laplacian plus a PD block diagonal -- stiff in ``delta``, and
    rotational wherever the skew parts of ``Gamma`` and ``Theta`` dominate, which is the regime
    ``EGLoss`` exists for. And ``Psi`` depends on the prices partly through the clamp in
    ``PriceSPE.flows``, whose derivative is zero on every closed route; the surrogate never
    differentiates through it, so a route that ought to open still exerts force and the model needs
    no straight-through relu (which is what ``FlowReadout`` has to carry).

    Nothing is normalized. The instance data is already O(1)-O(10) in comparable units (intercepts
    1-8, route costs 0.5-3, effect matrices at the sampler's ``scale``), the prediction is in raw
    price units, and the loss evaluates the true operator -- so ``train/loss`` and ``val/residual``
    are both in ground-truth units and comparable across runs.

    Args:
        backbone: ``SpeTransformerBackbone``, called with the five feature tensors.
        dim: backbone output width, the readout's input width.
        loss: a field surrogate from ``l2s_games.losses``, called as ``loss(field_fn, iterate)``.
            ``PotentialLoss`` gives one operator evaluation per step and converges on a strongly
            monotone instance; ``EGLoss`` costs two and handles a merely monotone one.
    """

    def __init__(
        self,
        backbone,
        dim,
        loss,
        lr=1e-3,
        start_factor=0.01,
        warmup_epochs=10,
        cosine_annealing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.backbone = backbone
        self.readout = PriceReadout(dim)
        self.loss = loss
        self.lr = lr
        self.start_factor = start_factor
        self.warmup_epochs = warmup_epochs
        self.cosine_annealing = cosine_annealing

    def predict_prices(self, batch):
        """The model's price vector per instance, ``[B, m + n]``, supply markets first."""
        supply, demand = self.backbone(
            batch["supply_feats"],
            batch["demand_feats"],
            batch["route_feats"],
            batch["supply_pairs"],
            batch["demand_pairs"],
        )
        return self.readout(supply, demand)

    def instance(self, batch):
        """The batch as one ``PriceSPE``: every field is a tensor with a leading batch dim, so the
        true operator evaluates the whole batch in one shot."""
        return PriceSPE.from_data(batch)

    def training_step(self, batch, batch_idx):
        instance = self.instance(batch)
        loss = self.loss(lambda prices: ascent_field(instance, prices), self.predict_prices(batch))
        # epoch-level: an epoch is a handful of steps, so the per-step default is noisy and
        # CSVLogger's 100-step flush would drop it
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        instance = self.instance(batch)
        prices = self.predict_prices(batch)
        # stationarity of the predicted prices under the true operator, in ground-truth units: zero
        # exactly at equilibrium, and unlike val/loss it does not change meaning with the surrogate,
        # so it stays comparable between the PotentialLoss and EGLoss runs
        self.log("val/residual", instance.operator(prices).norm(dim=-1).mean())
        self.log("val/loss", self.loss(lambda iterate: ascent_field(instance, iterate), prices))
        # the active set is the combinatorial content of the problem, and the part the price head
        # only reaches indirectly: prices that open no routes have collapsed onto the market-clearing
        # terms alone, which this catches and the residual alone would not
        self.log("val/open_routes", (instance.flows(prices) > 0).float().mean())
        return prices

    def configure_optimizers(self):
        # Adam with linear warmup then optional cosine annealing: warmup ramps from lr*start_factor
        # up to lr over warmup_epochs, then cosine decays to ~0 over the rest. Requires
        # warmup_epochs < trainer.max_epochs (else the cosine T_max is non-positive).
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=self.start_factor, total_iters=self.warmup_epochs
        )
        if self.cosine_annealing:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.trainer.max_epochs - self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup, cosine], milestones=[self.warmup_epochs])
        else:
            scheduler = warmup
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
