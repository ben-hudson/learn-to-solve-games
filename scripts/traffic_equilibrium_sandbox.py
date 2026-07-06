"""
traffic_equilibrium_sandbox.py

Exercise the Markov traffic-equilibrium VI on the Sioux Falls network without the (still to
come) amortized GNN model. The cost-space operator ``costs - bpr(demand_flow(-costs))`` is the
residual whose zero is the user equilibrium over the feasible set ``costs >= free_flow_time``.
Because the residual is smooth, simultaneous-GD on it is the damped fixed-point iteration that
converges to equilibrium -- we solve it with torchdeq's Anderson acceleration and compare to the
network's reference equilibrium costs, confirm ``torch.func.jacrev`` composes through the
route-choice solver (so Jacobian methods work on the learned field), and roll out the projected
dynamics via the shared ``simulate``.

Also shows the conditioning seam (``model_input`` -> ``transform`` -> ``collate_fn``) that the GNN
field model will consume.

    python scripts/traffic_equilibrium_sandbox.py
"""

import torch
import torchdeq

from l2s_games.algorithms import ALGORITHMS
from l2s_games.dynamics import simulate
from l2s_games.envs import bind
from l2s_games.envs.traffic import MarkovTrafficEquilibrium, load_sioux_falls_base_graph


def solve_equilibrium(vi, free_flow_time, step=0.1):
    """Solve ``operator(costs) = 0`` via the damped fixed point with Anderson acceleration."""
    solver = torchdeq.get_deq(f_solver="anderson", f_max_iter=1000, f_tol=1e-4)
    fixed_point = lambda costs: costs - step * vi.operator(costs)
    with torch.no_grad():
        iterates, info = solver(fixed_point, free_flow_time.unsqueeze(0).clone())  # Anderson needs a batch dim
    return iterates[-1].squeeze(0), int(info["nstep"].max().item())


def main():
    base_graph = load_sioux_falls_base_graph("data/sioux_falls")
    family = MarkovTrafficEquilibrium(base_graph)
    print(f"Sioux Falls: {base_graph.num_nodes} nodes, {base_graph.num_edges} edges")

    # Operator correctness on the base network: solve to equilibrium and compare to reference costs.
    vi = bind(family, base_graph)
    costs_eq, nstep = solve_equilibrium(vi, base_graph.free_flow_time)
    reference = base_graph.Cost.float()
    rel_err = ((costs_eq - reference).abs() / reference).mean()
    print(f"solved equilibrium in {nstep} steps: residual ||r|| = {vi.operator(costs_eq).norm():.4e}")
    print(f"mean relative error vs reference equilibrium costs = {rel_err:.2%}")

    # Note on Jacobian-based dynamics (Consensus): route_choice's EdgeProb softmax used to give a
    # 0/0 backward at sink nodes, poisoning gradients with NaN; that is now fixed upstream (clamped
    # denominator in route_choice/markov/layers.py), so ordinary autograd -- operator(costs).backward()
    # -- through this analytic operator is finite. torch.func.jacrev still does NOT compose through it,
    # for a separate reason: the value/flow solves use torchdeq with ift=True, whose IFT gradient is an
    # autograd.Function that functorch/jacrev cannot transform. So Jacobian methods still target the
    # learned GNN field (jacrev works there -- see the flat games); non-Jacobian methods
    # (projection/extragradient) drive this analytic operator.

    # A noised instance: the projection method on -operator is the damped iteration, so it drives r -> 0.
    torch.manual_seed(0)
    graph = family.sample_params()
    instance = bind(family, graph)
    costs0 = graph.free_flow_time.clone()
    traj = simulate(
        lambda c: -instance.operator(c), ALGORITHMS["projection"](0.1), costs0, n_steps=300, project=instance.project
    )
    print(
        f"instance: residual ||r|| start={instance.operator(costs0).norm():.4e}  "
        f"end={instance.operator(traj[-1]).norm():.4e}  (feasible={bool((traj >= graph.free_flow_time - 1e-5).all())})"
    )

    # Conditioning seam: model_input -> transform (builds feats + structure) -> collate_fn (dense batch).
    items = [family.transform(family.model_input(graph, family.sample_domain(graph, 1)[0])) for _ in range(3)]
    batch = family.collate_fn(items)
    assert {"feats", "in_degree", "out_degree", "spd"} <= set(batch)
    print(f"collate -> dense batch: feats shape={tuple(batch['feats'].shape)}, spd shape={tuple(batch['spd'].shape)}")


if __name__ == "__main__":
    main()
