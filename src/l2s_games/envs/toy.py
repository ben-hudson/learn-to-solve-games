import torch

from l2s_games.envs.base import Game, ParamSpec


def rotational_field(omega=1.0, damp_floor=0.0, damp_wall=0.0, well_angle=-45.0, curl_nonlin=0.0):
    """Parametric field = oriented anisotropic potential well + rotation.

    The linear part is  A = S + R,  where R = omega * [[0,-1],[1,0]] is the
    antisymmetric (rotational) part and S is a symmetric well:

        S = -damp_floor * u u^T  -  damp_wall * w w^T

    with u = (cos a, sin a) the trough-floor direction and w its perpendicular
    (a = well_angle). damp_wall contracts across the trough (steep walls),
    damp_floor along it (a tilted floor). Equal floor/wall = a round bowl, which
    recovers the isotropic spiral; damp_floor=0 with damp_wall>0 gives a trough.

    Works on a single point (shape (2,)) or a grid (shape (..., 2)).
    """
    a = torch.deg2rad(torch.tensor(well_angle))
    u = torch.tensor([torch.cos(a), torch.sin(a)])
    w = torch.tensor([-torch.sin(a), torch.cos(a)])
    S = -damp_floor * torch.outer(u, u) - damp_wall * torch.outer(w, w)
    R = omega * torch.tensor([[0.0, -1.0], [1.0, 0.0]])

    def v(z):
        z = torch.as_tensor(z, dtype=torch.float32)
        th, ps = z[..., 0], z[..., 1]
        # rotation is scaled by the Dirac-GAN-style nonlinearity; the well is linear
        rot = 1.0 + curl_nonlin * th * ps
        v_rot_th = rot * (R[0, 0] * th + R[0, 1] * ps)
        v_rot_ps = rot * (R[1, 0] * th + R[1, 1] * ps)
        v_well_th = S[0, 0] * th + S[0, 1] * ps
        v_well_ps = S[1, 0] * th + S[1, 1] * ps
        return torch.stack([v_rot_th + v_well_th, v_rot_ps + v_well_ps], dim=-1)

    return v


class RotationalFieldGame(Game):
    """The ``rotational_field`` family as a game: (omega, damp_floor, damp_wall)."""

    def __init__(self, lim=2.0, well_angle=45.0, curl_nonlin=0.0, ranges=((0.0, 1.0),) * 3):
        self.lim = lim
        self.well_angle = well_angle
        self.curl_nonlin = curl_nonlin
        self._ranges = ranges

    @property
    def domain_dim(self):
        return 2

    @property
    def param_specs(self):
        names = ("omega", "damp_floor", "damp_wall")
        return tuple(ParamSpec(name, low, high) for name, (low, high) in zip(names, self._ranges))

    def operator(self, params, points):
        omega, damp_floor, damp_wall = params
        field = rotational_field(omega, damp_floor, damp_wall, self.well_angle, self.curl_nonlin)
        return field(points)

    def sample_points(self, n):
        return (2 * torch.rand(n, self.domain_dim) - 1) * self.lim
