import dataclasses
from typing import NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.repr_base import RBC
from genesis.utils.array_class import DataKind


@dataclasses.dataclass
class SolverCheckpoint:
    """The arrays of one solver of the given kinds (see 'DataKind'), for 'Solver.__setstate__' to write back.

    A checkpoint '__getstate__' reads holds every kind a step reads from one step to the next (see 'CHECKPOINT_KINDS'),
    so stepping on from it lands where the original went. A trajectory frame may hold the state alone, and writing it
    back recomputes what it leaves out. Each array stands under the name 'Solver.data' gives it, copied where it lives:
    on the device as a torch tensor when zero-copy views exist, as a numpy array otherwise. The static configs come
    along as provenance and are never written back.
    """

    arrays: dict[str, np.ndarray | torch.Tensor]
    configs: dict[str, object]
    kinds: frozenset[DataKind]


@dataclasses.dataclass
class KinematicSolverCheckpoint(SolverCheckpoint):
    """A SolverCheckpoint plus the two forward-kinematics flags of a kinematic or rigid solver,
    'is_forward_pos_updated' and 'is_forward_vel_updated'.

    The flags decide whether the next step recomputes the Cartesian pose and velocity of the links. Recomputing them
    where the saved scene skipped it rounds differently on some backends, so the flags travel with the derived arrays
    they describe.
    """

    is_forward_pos_updated: bool
    is_forward_vel_updated: bool


class FrameField(NamedTuple):
    """Where one array stands in a trajectory frame.

    The name, canonical shape, numpy dtype string, byte offset and byte length cut a frame back into its arrays without
    the scene.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int


@dataclasses.dataclass
class SimulatorCheckpoint:
    """The state a simulation stands in.

    How many steps each environment has run, and what every active solver reads from one step to the next, under the
    name of its class.
    """

    steps: np.ndarray | torch.Tensor
    solvers: dict[str, SolverCheckpoint]


class SimState(RBC):
    """
    Dynamic state queried from a Scene's Simulator.
    """

    def __init__(
        self,
        scene,
        s_global,
        f_local,
        solvers,
    ):
        self._scene = scene
        self._s_global = s_global
        self._solvers_state = list()
        for solver in solvers:
            self._solvers_state.append(solver.get_state(f_local))

    @property
    def scene(self):
        return self._scene

    @property
    def s_global(self):
        return self._s_global

    @property
    def solvers_state(self):
        return self._solvers_state

    def __iter__(self):
        return iter(self._solvers_state)


class KinematicSolverState:
    """
    Dynamic state queried from a KinematicSolver.

    Only stores position-related fields (qpos, link poses). Physics fields
    (velocity, acceleration, mass, friction) are omitted since kinematic entities have no dynamics.
    """

    def __init__(self, scene, s_global):
        self.scene = scene
        self._s_global = s_global

        _B = scene.sim.kinematic_solver._B
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self.scene,
        }
        self.qpos = gs.zeros((_B, scene.sim.kinematic_solver.n_qs), **args)
        self.dofs_vel = gs.zeros((_B, scene.sim.kinematic_solver.n_dofs), **args)
        self.links_pos = gs.zeros((_B, scene.sim.kinematic_solver.n_links, 3), **args)
        self.links_quat = gs.zeros((_B, scene.sim.kinematic_solver.n_links, 4), **args)

    @property
    def s_global(self):
        return self._s_global


class RigidSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene, s_global):
        self.scene = scene

        self._s_global = s_global

        _B = scene.sim.rigid_solver._B
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self.scene,
        }
        self.qpos = gs.zeros((_B, scene.sim.rigid_solver.n_qs), **args)
        self.dofs_vel = gs.zeros((_B, scene.sim.rigid_solver.n_dofs), **args)
        self.dofs_acc = gs.zeros((_B, scene.sim.rigid_solver.n_dofs), **args)
        self.links_pos = gs.zeros((_B, scene.sim.rigid_solver.n_links, 3), **args)
        self.links_quat = gs.zeros((_B, scene.sim.rigid_solver.n_links, 4), **args)
        self.friction_ratio = gs.ones((_B, scene.sim.rigid_solver.n_geoms), **args)

    @property
    def s_global(self):
        return self._s_global


class ToolSolverState:
    """
    Dynamic state queried from a RigidSolver.
    """

    def __init__(self, scene):
        self.scene = scene
        self.entities = []

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        return self.entities[index]

    # def __repr__(self):
    #     return f'{_repr(self)}\n' \
    #            f'entities : {_repr(self.entities)}'


class MPMSolverState(RBC):
    """
    Dynamic state queried from a MPMSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3), **args)
        self._C = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3, 3), **args)
        self._F = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles, 3, 3), **args)
        self._Jp = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((scene.sim._B, scene.sim.mpm_solver.n_particles), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHSolverState:
    """
    Dynamic state queried from a SPHSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.sph_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.sph_solver.n_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((self._scene.sim._B, scene.sim.sph_solver.n_particles), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class PBDSolverState:
    """
    Dynamic state queried from a PBDSolver.
    """

    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.pbd_solver.n_particles, 3), **args)
        self._vel = gs.zeros((self._scene.sim._B, scene.sim.pbd_solver.n_particles, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._free = gs.zeros((self._scene.sim._B, scene.sim.pbd_solver.n_particles), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def free(self):
        return self._free


class FEMSolverState:
    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }
        self._pos = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_vertices, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_vertices, 3), **args)
        args["dtype"] = gs.tc_bool
        args["requires_grad"] = False
        self._active = gs.zeros((scene.sim._B, scene.sim.fem_solver.n_elements), **args)

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class RODSolverState:
    def __init__(self, scene):
        self._scene = scene
        args = {
            "dtype": gs.tc_float,
            "requires_grad": scene.requires_grad,
            "scene": self._scene,
        }

        # vertex states
        self._pos = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_vertices, 3), **args)
        self._vel = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_vertices, 3), **args)

        # edge states
        self._edge = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges, 3), **args)
        self._length = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges), **args)
        self._d1 = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges, 3), **args)
        self._d2 = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges, 3), **args)
        self._d3 = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges, 3), **args)
        self._d1_ref = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges, 3), **args)
        self._d2_ref = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges, 3), **args)
        self._theta = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges), **args)
        self._omega = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_edges), **args)

        # internal vertex states
        self._kb = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_internal_vertices, 3), **args)
        self._twist = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_internal_vertices), **args)
        self._kappa_rest = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_internal_vertices, 2), **args)

        args["requires_grad"] = False
        args["dtype"] = gs.tc_bool
        # no grad, bool
        self._fixed = gs.zeros((scene.sim._B, scene.sim.rod_solver.n_vertices), **args)

    def serializable(self):
        self._scene = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._fixed = self._fixed.detach()

        self._edge = self._edge.detach()
        self._length = self._length.detach()
        self._d1 = self._d1.detach()
        self._d2 = self._d2.detach()
        self._d3 = self._d3.detach()
        self._d1_ref = self._d1_ref.detach()
        self._d2_ref = self._d2_ref.detach()
        self._theta = self._theta.detach()
        self._omega = self._omega.detach()

        self._kb = self._kb.detach()
        self._twist = self._twist.detach()
        self._kappa_rest = self._kappa_rest.detach()

    @property
    def scene(self):
        return self._scene

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def fixed(self):
        return self._fixed

    @property
    def edge(self):
        return self._edge

    @property
    def length(self):
        return self._length

    @property
    def d1(self):
        return self._d1

    @property
    def d2(self):
        return self._d2

    @property
    def d3(self):
        return self._d3

    @property
    def d1_ref(self):
        return self._d1_ref

    @property
    def d2_ref(self):
        return self._d2_ref

    @property
    def theta(self):
        return self._theta

    @property
    def omega(self):
        return self._omega

    @property
    def kb(self):
        return self._kb

    @property
    def twist(self):
        return self._twist

    @property
    def kappa_rest(self):
        return self._kappa_rest
