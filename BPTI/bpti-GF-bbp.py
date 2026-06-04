#!/usr/bin/env python
# coding: utf-8

import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import openmm
from openmm import unit

import bgflow as bg
import bgmol
from bgflow.utils.types import assert_numpy

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as TorchDDP
except ImportError:
    dist = None
    TorchDDP = None

from bgmol.datasets.base import DataSet
import mdtraj as md
from bgmol.systems.base import OpenMMSystem
from bgmol.util.importing import import_openmm
from bgmol.zmatrix import ZMatrixFactory, build_fake_topology

from bgflow import (
    TORSIONS,
    FIXED,
    BONDS,
    ANGLES,
    ShapeDictionary,
    InternalCoordinateMarginals,
)

from bgflow.distribution.sampling import DataSetSampler
from bgmol import bond_constraints


PROJECT_DIR = Path(__file__).resolve().parent
TOP_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_nowat_align.pdb"
TRAJ_T500_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_traj_nowat_align.dcd"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_WORLD_SIZE = int(os.environ.get("REQUIRED_WORLD_SIZE", "4"))


def init_distributed(required_world_size=2):
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size_env > 1

    if is_distributed:
        if dist is None or TorchDDP is None:
            raise RuntimeError("torch.distributed and DistributedDataParallel are required for multi-GPU training")
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA GPUs")
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Please launch with torchrun so LOCAL_RANK is set")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if required_world_size is not None and world_size != required_world_size:
            raise RuntimeError(
                f"This script expects WORLD_SIZE={required_world_size}, got {world_size}. "
                f"Use: torchrun --nproc_per_node={required_world_size} {Path(__file__).name}"
            )

        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        if required_world_size is not None and required_world_size > 1:
            print(
                f"[WARN] Running single process (WORLD_SIZE=1). "
                f"For 2-GPU training use: torchrun --nproc_per_node={required_world_size} {Path(__file__).name}",
                flush=True,
            )

    return {
        "is_distributed": is_distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
        "device": device,
    }


def cleanup_distributed():
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_barrier():
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank_print(msg, *, force=False):
    if force or DDP["is_main"]:
        print(msg, flush=True)


def set_seed(base_seed=20260529):
    seed = base_seed + DDP["rank"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shard_tensor(data, world_size, rank):
    n = data.shape[0]
    base = n // world_size
    remainder = n % world_size
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    if start >= stop:
        raise RuntimeError(
            f"Rank {rank} received empty shard (n={n}, world_size={world_size}); reduce world size or use more data"
        )
    return data[start:stop]


def local_batch_size(global_batch_size):
    if global_batch_size < 1:
        raise ValueError("global_batch_size must be >= 1")
    return max(1, global_batch_size // DDP["world_size"])


def unwrap_module(module):
    return module.module if isinstance(module, TorchDDP) else module


def select_ca_fixed_atoms_from_secondary_structure(trajectory, ca_atom_indices):
    if len(ca_atom_indices) < 3:
        raise ValueError("CA fixed relative coordinate transform requires at least 3 CA atoms")

    dssp = md.compute_dssp(trajectory[0], simplified=True)[0]
    ca_to_local = {}
    for local_idx, atom_idx in enumerate(ca_atom_indices):
        atom = trajectory.topology.atom(int(atom_idx))
        ca_to_local[atom.residue.index] = local_idx

    ca_labels = []
    for residue_idx, ss in enumerate(dssp):
        ca_idx = ca_to_local.get(residue_idx)
        if ca_idx is None:
            continue
        ca_labels.append((ca_idx, ss))

    fixed_atoms = []
    if ca_labels:
        segment = [ca_labels[0][0]]
        segment_label = ca_labels[0][1]
        segments = []
        for ca_idx, ss in ca_labels[1:]:
            if ss == segment_label and ca_idx == segment[-1] + 1:
                segment.append(ca_idx)
            else:
                segments.append((segment_label, segment))
                segment = [ca_idx]
                segment_label = ss
        segments.append((segment_label, segment))

        for ss, segment in segments:
            if ss == "C":
                fixed_atoms.extend(segment[::2])
            else:
                fixed_atoms.append(segment[len(segment) // 2])

    fixed_atoms = sorted(set(fixed_atoms))
    if len(fixed_atoms) < 3:
        missing = [idx for idx in range(len(ca_atom_indices)) if idx not in fixed_atoms]
        fixed_atoms.extend(missing[: 3 - len(fixed_atoms)])

    return sorted(fixed_atoms)


def maybe_wrap_ddp(module, find_unused_parameters=False):
    if DDP["is_distributed"]:
        return TorchDDP(
            module,
            device_ids=[DDP["local_rank"]],
            output_device=DDP["local_rank"],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    return module


class DistributedOptimizerWrapper:
    def __init__(self, optimizer, params):
        self.optimizer = optimizer
        self.params = list(params)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        if DDP["is_distributed"] and dist is not None and dist.is_initialized():
            for p in self.params:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= DDP["world_size"]
        return self.optimizer.step(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


DDP = init_distributed(required_world_size=REQUIRED_WORLD_SIZE)
ctx = {"dtype": torch.float32, "device": DDP["device"]}
kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
set_seed(20260529)
rank_print(f"[Rank {DDP['rank']}] device={ctx['device']} world_size={DDP['world_size']}", force=True)

_, unit, app = import_openmm()


class BPTIImplicit(OpenMMSystem):
    def __init__(
        self,
        constraints=app.HBonds,
        hydrogen_mass=4.0 * unit.amu,
    ):
        super(BPTIImplicit, self).__init__()

        self.constraints = self.system_parameter("constraints", constraints, default=app.HBonds)
        self.hydrogen_mass = self.system_parameter("hydrogen_mass", hydrogen_mass, default=4.0 * unit.amu)

        forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        pdb = app.PDBFile(str(TOP_FILE))

        self._system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=self.constraints,
            hydrogenMass=self.hydrogen_mass,
        )

        self._positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        self._topology = pdb.topology


class BPTIImplicitTlow(DataSet):
    def __init__(self, root=os.getcwd(), download: bool = False, read: bool = False):
        super(BPTIImplicitTlow, self).__init__(root=root, download=download, read=read)
        self._system = BPTIImplicit()
        self._temperature = 300

    @property
    def top_file(self):
        return str(TOP_FILE)

    def read(self, n_frames=None, stride=None, atom_indices=None):
        self.trajectory = md.load(self.top_file, atom_indices=atom_indices)


class BPTIImplicitThigh(DataSet):
    def __init__(self, root=os.getcwd(), download: bool = False, read: bool = False):
        super(BPTIImplicitThigh, self).__init__(root=root, download=download, read=read)
        self._system = BPTIImplicit()
        self._temperature = 500

    @property
    def trajectory_file(self):
        return str(TRAJ_T500_FILE)

    @property
    def top_file(self):
        return str(TOP_FILE)

    def read(self, n_frames=None, stride=None, atom_indices=None):
        self.trajectory = md.load(self.trajectory_file, top=self.top_file, stride=stride, atom_indices=atom_indices)
        if n_frames is not None:
            self.trajectory = self.trajectory[:n_frames]


def plot_report(trainer, title_prefix=""):
    if not DDP["is_main"]:
        return
    for i, label in enumerate(trainer.reporter._labels):
        plt.figure()
        plt.plot(trainer.reporter._raw[i])
        plt.title(f"{title_prefix}{label}")
        plt.tight_layout()
        plt.show()


def get_phi_psi(trajectory, model):
    if not isinstance(trajectory, md.Trajectory):
        if isinstance(trajectory, torch.Tensor):
            trajectory = assert_numpy(trajectory.view(len(trajectory), *model.positions.shape))
        trajectory = md.Trajectory(trajectory, model.mdtraj_topology)
    phi_atoms_list = md.compute_phi(trajectory)[0]
    psi_atoms_list = md.compute_psi(trajectory)[0]
    phi = md.compute_dihedrals(trajectory, indices=phi_atoms_list)
    psi = md.compute_dihedrals(trajectory, indices=psi_atoms_list)
    return phi, psi


def run():
    is_data_here = True
    dataset_1 = BPTIImplicitThigh(download=(not is_data_here), read=True)
    system_1 = dataset_1.system
    coordinates_1 = dataset_1.xyz

    rank_print(f"dataset xyz shape: {coordinates_1.shape}", force=True)

    target_energy_1 = dataset_1.get_energy_model(n_workers=16)

    data_len = len(coordinates_1)
    g = torch.Generator()
    g.manual_seed(20260529)
    indices = torch.randperm(data_len, generator=g)

    len_training = int(data_len * 0.8)
    training_indices = indices[:len_training]
    testing_indices = indices[len_training:]

    training_full = torch.tensor(coordinates_1[training_indices], dtype=ctx["dtype"])
    testing_full = torch.tensor(coordinates_1[testing_indices], dtype=ctx["dtype"])

    if DDP["is_distributed"]:
        training_full = shard_tensor(training_full, DDP["world_size"], DDP["rank"])
        testing_full = shard_tensor(testing_full, DDP["world_size"], DDP["rank"])

    training_data_gen = training_full.to(device=ctx["device"], non_blocking=True)
    test_data = testing_full.to(device=ctx["device"], non_blocking=True)

    rank_print(f"[Rank {DDP['rank']}] train shard: {tuple(training_data_gen.shape)}", force=True)
    rank_print(f"[Rank {DDP['rank']}] test shard: {tuple(test_data.shape)}", force=True)

    c_alpha = system_1.mdtraj_topology.select("name CA")

    zfactory = ZMatrixFactory(system_1.mdtraj_topology, cartesian=c_alpha)
    z_matrix, fixed_atoms = zfactory.build_naive()

    coordinate_transform = bg.RelativeInternalCoordinateTransformation(
        z_matrix=z_matrix,
        fixed_atoms=fixed_atoms,
        normalize_angles=True,
    ).to(**ctx)

    shape_info = bg.ShapeDictionary.from_coordinate_transform(
        coordinate_transform,
        n_constraints=system_1.system.getNumConstraints(),
    )

    ca_fixed_atoms = select_ca_fixed_atoms_from_secondary_structure(dataset_1.trajectory, c_alpha)
    dssp = md.compute_dssp(dataset_1.trajectory[0], simplified=True)[0]
    ca_fixed_summary = []
    for ca_idx in ca_fixed_atoms:
        atom = dataset_1.trajectory.topology.atom(int(c_alpha[ca_idx]))
        residue = atom.residue
        ca_fixed_summary.append(
            f"ca_idx={ca_idx}, atom={int(c_alpha[ca_idx])}, residue={residue}, ss={dssp[residue.index]}"
        )
    rank_print("Selected CA fixed atoms:\n  " + "\n  ".join(ca_fixed_summary))

    top, _ = build_fake_topology(len(fixed_atoms))
    ca_zfactory = ZMatrixFactory(top, cartesian=ca_fixed_atoms)
    ca_z_matrix, _ = ca_zfactory.build_naive()

    ca_crd_transform = bg.RelativeInternalCoordinateTransformation(
        z_matrix=ca_z_matrix,
        fixed_atoms=ca_fixed_atoms,
        normalize_angles=True,
    ).to(**ctx)
    ca_shape_info = ShapeDictionary.from_coordinate_transform(ca_crd_transform)
    CA_BONDS = ca_shape_info.replace(BONDS, "CA_BONDS")
    CA_ANGLES = ca_shape_info.replace(ANGLES, "CA_ANGLES")
    CA_TORSIONS = ca_shape_info.replace(TORSIONS, "CA_TORSIONS")
    CA_FIXED = ca_shape_info.replace(FIXED, "CA_FIXED")

    shape_info.update(ca_shape_info)
    del shape_info[FIXED]

    ca_fixed_shapes = ShapeDictionary()
    ca_fixed_shapes[CA_FIXED] = ca_shape_info[CA_FIXED]
    ca_fixed_builder = bg.BoltzmannGeneratorBuilder(ca_fixed_shapes, **ctx)
    caf1, caf2 = ca_fixed_builder.add_split(
        CA_FIXED,
        into=["CAF1", "CAF2"],
        sizes_or_indices=[
            np.arange(0, ca_shape_info[CA_FIXED][0], 2),
            np.arange(1, ca_shape_info[CA_FIXED][0], 2),
        ],
    )
    for _ in range(6):
        ca_fixed_builder.add_condition(caf1, on=caf2)
        ca_fixed_builder.add_condition(caf2, on=caf1)
    ca_fixed_builder.add_merge(
        (caf1, caf2),
        to=CA_FIXED,
        sizes_or_indices=[
            np.arange(0, ca_shape_info[CA_FIXED][0], 2),
            np.arange(1, ca_shape_info[CA_FIXED][0], 2),
        ],
    )
    ca_fixed_generator = ca_fixed_builder.build_generator().to(**ctx)

    ca_torsion_shapes = ShapeDictionary()
    ca_torsion_shapes[CA_TORSIONS] = ca_shape_info[CA_TORSIONS]
    ca_torsion_shapes[CA_FIXED] = ca_shape_info[CA_FIXED]
    ca_torsion_builder = bg.BoltzmannGeneratorBuilder(ca_torsion_shapes, **ctx)
    ca_torsion_builder.add_layer(ca_fixed_generator._flow, what=(CA_FIXED,))
    cat1, cat2 = ca_torsion_builder.add_split(
        CA_TORSIONS,
        into=["CAT1", "CAT2"],
        sizes_or_indices=[
            np.arange(0, ca_shape_info[CA_TORSIONS][0], 2),
            np.arange(1, ca_shape_info[CA_TORSIONS][0], 2),
        ],
    )
    for _ in range(6):
        ca_torsion_builder.add_condition(cat1, on=(cat2, CA_FIXED))
        ca_torsion_builder.add_condition(cat2, on=(cat1, CA_FIXED))
    ca_torsion_builder.add_merge(
        (cat1, cat2),
        to=CA_TORSIONS,
        sizes_or_indices=[
            np.arange(0, ca_shape_info[CA_TORSIONS][0], 2),
            np.arange(1, ca_shape_info[CA_TORSIONS][0], 2),
        ],
    )
    ca_torsion_generator = ca_torsion_builder.build_generator().to(**ctx)

    ca_full_builder = bg.BoltzmannGeneratorBuilder(ca_shape_info, **ctx)
    ca_full_builder.add_layer(ca_torsion_generator._flow, what=(CA_TORSIONS, CA_FIXED))

    caa1, caa2 = ca_full_builder.add_split(
        CA_ANGLES,
        into=["CAA1", "CAA2"],
        sizes_or_indices=[
            np.arange(0, ca_shape_info[CA_ANGLES][0], 2),
            np.arange(1, ca_shape_info[CA_ANGLES][0], 2),
        ],
    )
    for _ in range(6):
        ca_full_builder.add_condition(caa1, on=(CA_TORSIONS, CA_BONDS, caa2), param_groups=("ca_ba",))
        ca_full_builder.add_condition(caa2, on=(CA_TORSIONS, CA_BONDS, caa1), param_groups=("ca_ba",))
    ca_full_builder.add_merge(
        (caa1, caa2),
        to=CA_ANGLES,
        sizes_or_indices=[
            np.arange(0, ca_shape_info[CA_ANGLES][0], 2),
            np.arange(1, ca_shape_info[CA_ANGLES][0], 2),
        ],
    )
    ca_full_builder.add_condition(CA_BONDS, on=(CA_TORSIONS, CA_ANGLES), param_groups=("ca_ba",))
    ca_ic_generator = ca_full_builder.build_generator()

    builder = bg.BoltzmannGeneratorBuilder(shape_info, target_energy_1, **ctx)
    builder.add_layer(ca_ic_generator._flow, [CA_BONDS, CA_ANGLES, CA_TORSIONS, CA_FIXED])

    conditioner = "allegro"
    if conditioner == "allegro":
        from bgflow.factory.GNN_factory import allegro_hparams as hparams, make_allegro_config_dict as make_config_dict
    elif conditioner in ["nequip", "schnet"]:
        from bgflow.factory.GNN_factory import nequip_hparams as hparams, make_nequip_config_dict as make_config_dict
    else:
        raise ValueError(f"Unsupported conditioner: {conditioner}")

    from bgflow.nn.periodic import WrapDistances

    distances_net = WrapDistances(torch.nn.Identity())
    fixed = coordinate_transform.forward(training_data_gen[:1000])[3]
    RBF_distances = distances_net(fixed.detach()).flatten().cpu()
    RBF_distances = RBF_distances[RBF_distances <= hparams["r_max"]]

    avg_num_neighbors = ((len(RBF_distances) / (1000 * fixed.shape[1] / 3))) * 2

    layers = make_config_dict(**hparams)
    layers["radial_basis"][1]["basis_kwargs"]["data"] = RBF_distances

    if conditioner == "allegro":
        layers["allegro"][1]["avg_num_neighbors"] = avg_num_neighbors

    if conditioner in ["nequip", "schnet"]:
        for i in range(hparams["num_interaction_blocks"]):
            layers[f"convnet_{i}"][1]["convolution_kwargs"]["avg_num_neighbors"] = avg_num_neighbors
        if conditioner == "schnet":
            hparams["irreps_edge_sh"] = "1x0e"

    from nequip.nn import SequentialGraphNetwork
    from bgflow.factory.GNN_factory import NequipWrapper

    GNN = SequentialGraphNetwork.from_parameters(shared_params=None, layers=layers)
    GNN_feature_extractor = NequipWrapper(GNN, cutoff=hparams["r_max"])

    builder.default_conditioner_type = "GNN"
    builder.default_conditioner_kwargs = {
        "r_max": hparams["r_max"],
        "GNN": GNN_feature_extractor,
        "use_checkpointing": False,
        "GNN_output_dim": hparams["GNN_feature_dim"] * len(c_alpha),
        "attention_units": len(c_alpha),
        "attention_level": "MHA",
    }

    CA_REP = [FIXED]

    ca_cdfs = InternalCoordinateMarginals(
        builder.current_dims,
        builder.ctx,
        bonds=CA_BONDS,
        angles=CA_ANGLES,
        torsions=CA_TORSIONS,
        fixed=CA_FIXED,
        bond_upper=3.0,
        bond_lower=0.03,
        angle_lower=0.1,
        angle_upper=1.0,
    )
    ca_cdfs.inform_with_data(
        torch.tensor(dataset_1.xyz[:, c_alpha, :], **ctx).reshape(-1, len(c_alpha) * 3),
        ca_crd_transform,
        bonds=CA_BONDS,
        angles=None,
        torsions=None,
    )

    ca_icdf_maps = builder.add_map_to_ic_domains(ca_cdfs, return_layers=True)
    builder.add_map_to_cartesian(
        ca_crd_transform,
        bonds=CA_BONDS,
        angles=CA_ANGLES,
        torsions=CA_TORSIONS,
        fixed=CA_FIXED,
        out=FIXED,
    )

    n_torsions = builder.current_dims[TORSIONS][-1]
    t1, t2 = builder.add_split(
        TORSIONS,
        into=["T1", "T2"],
        sizes_or_indices=[np.arange(0, n_torsions, 2), np.arange(1, n_torsions, 2)],
    )
    for _ in range(4):
        builder.add_condition(t1, on=(*CA_REP, t2), hidden=(512, 512), param_groups=("aa", "torsions"))
        builder.add_condition(t2, on=(t1, *CA_REP), hidden=(512, 512), param_groups=("aa", "torsions"))
    builder.add_merge(
        (t1, t2),
        to=TORSIONS,
        sizes_or_indices=[np.arange(0, n_torsions, 2), np.arange(1, n_torsions, 2)],
    )

    for _ in range(2):
        builder.add_condition(BONDS, on=(ANGLES, *CA_REP, TORSIONS), hidden=(512, 512), param_groups=("aa",))
        builder.add_condition(ANGLES, on=(BONDS, TORSIONS, *CA_REP), hidden=(512, 512), param_groups=("aa",))

    cdfs = InternalCoordinateMarginals(
        builder.current_dims,
        builder.ctx,
        bonds=BONDS,
        angles=ANGLES,
        torsions=None,
        fixed=None,
        bond_mu=0.2,
        bond_sigma=2.0,
        bond_upper=1.0,
        bond_lower=0.03,
        angle_lower=0.001,
        angle_upper=0.9,
    )

    cdfs.inform_with_data(
        torch.tensor(dataset_1.xyz[::2000], **ctx).reshape(len(dataset_1.xyz[::2000]), -1),
        coordinate_transform,
        constrained_bond_indices=bond_constraints(system_1.system, coordinate_transform)[0],
        torsions=None,
    )
    _ = builder.add_map_to_ic_domains(cdfs, return_layers=True)
    builder.add_merge_constraints(*bond_constraints(system_1.system, coordinate_transform))

    builder.add_map_to_cartesian(coordinate_transform)
    generator = builder.build_generator(zero_parameters=False).to(**ctx)

    with torch.no_grad():
        _, _, _, ca, _ = coordinate_transform.forward(training_data_gen)
        ca_bonds, ca_angles, ca_torsions, ca_fixed, *_ = ca_crd_transform.forward(ca)
        ca_bonds = ca_icdf_maps[0].forward(ca_bonds, inverse=True)[0]
        ca_angles = ca_icdf_maps[1].forward(ca_angles, inverse=True)[0]
        ca_torsions = ca_icdf_maps[2].forward(ca_torsions, inverse=True)[0]
        ca_fixed = ca_icdf_maps[3].forward(ca_fixed, inverse=True)[0]

        _, _, _, tca, _ = coordinate_transform.forward(test_data[:10000])
        tca_bonds, tca_angles, tca_torsions, tca_fixed, *_ = ca_crd_transform.forward(tca)
        tca_bonds = ca_icdf_maps[0].forward(tca_bonds, inverse=True)[0]
        tca_angles = ca_icdf_maps[1].forward(tca_angles, inverse=True)[0]
        tca_torsions = ca_icdf_maps[2].forward(tca_torsions, inverse=True)[0]
        tca_fixed = ca_icdf_maps[3].forward(tca_fixed, inverse=True)[0]

    # Stage 1a: train CA fixed coordinates
    ca_fixed_params = list(ca_fixed_generator.parameters())
    ca_fixed_optimizer = DistributedOptimizerWrapper(
        torch.optim.Adam(ca_fixed_params, lr=1e-4),
        ca_fixed_params,
    )
    ca_fixed_trainer = bg.KLTrainer(
        ca_fixed_generator,
        optim=ca_fixed_optimizer,
        train_energy=False,
        test_likelihood=True,
    )

    ca_fixed_generator.train(True)
    ca_fixed_trainer.train(
        n_iter=1000,
        data=ca_fixed,
        testdata=tca_fixed,
        batchsize=local_batch_size(1024),
        n_print=100,
        w_energy=0.0,
    )
    ca_fixed_generator.train(False)

    maybe_barrier()
    ca_fixed_path = MODEL_DIR / "CA_FIXED_bpti_ddp.pic"
    if DDP["is_main"]:
        torch.save(ca_fixed_generator.state_dict(), ca_fixed_path)
    maybe_barrier()
    ca_fixed_generator.load_state_dict(torch.load(ca_fixed_path, map_location=ctx["device"], weights_only=True))

    # Stage 1b: train CA torsions
    ca_torsion_params = list(ca_torsion_generator.parameters())
    ca_torsion_optimizer = DistributedOptimizerWrapper(
        torch.optim.Adam(ca_torsion_params, lr=1e-4),
        ca_torsion_params,
    )
    ca_torsion_trainer = bg.KLTrainer(
        ca_torsion_generator,
        optim=ca_torsion_optimizer,
        train_energy=False,
        test_likelihood=True,
    )

    ca_torsion_generator.train(True)
    ca_torsion_trainer.train(
        n_iter=1000,
        data=DataSetSampler(ca_torsions, ca_fixed),
        testdata=DataSetSampler(tca_torsions, tca_fixed),
        batchsize=local_batch_size(1024),
        n_print=100,
        w_energy=0.0,
    )
    ca_torsion_generator.train(False)

    maybe_barrier()
    ca_torsion_path = MODEL_DIR / "CA_TORSIONS_bpti_ddp.pic"
    if DDP["is_main"]:
        torch.save(ca_torsion_generator.state_dict(), ca_torsion_path)
    maybe_barrier()
    ca_torsion_generator.load_state_dict(torch.load(ca_torsion_path, map_location=ctx["device"], weights_only=True))

    # Stage 2: train AA params only
    # Keep notebook behavior: optimize only builder.param_groups["aa"].
    aa_params = list(builder.param_groups["aa"])
    nll_optimizer = DistributedOptimizerWrapper(
        torch.optim.Adam(aa_params, lr=1e-4),
        aa_params,
    )
    nll_trainer = bg.KLTrainer(
        generator,
        optim=nll_optimizer,
        train_energy=False,
        test_likelihood=True,
    )

    generator.train(True)
    nll_trainer.train(
        n_iter=1000,
        data=training_data_gen,
        testdata=test_data,
        batchsize=local_batch_size(256),
        n_print=100,
        w_energy=0.0,
    )
    generator.train(False)

    maybe_barrier()
    nll_path = MODEL_DIR / "BPTI_1UAO_NLL_T500_ddp.pic"
    if DDP["is_main"]:
        torch.save(generator.state_dict(), nll_path)
    maybe_barrier()
    generator.load_state_dict(torch.load(nll_path, map_location=ctx["device"], weights_only=True))

    # Stage 3: full fine-tuning
    full_params = list(generator.parameters())
    full_nll_optimizer = DistributedOptimizerWrapper(
        torch.optim.Adam(full_params, lr=1e-4),
        full_params,
    )
    full_nll_trainer = bg.KLTrainer(
        generator,
        optim=full_nll_optimizer,
        train_energy=False,
        test_likelihood=True,
    )

    generator.train(True)
    full_nll_trainer.train(
        n_iter=1000,
        data=training_data_gen,
        testdata=test_data,
        batchsize=local_batch_size(256),
        n_print=100,
        w_energy=0.0,
    )
    generator.train(False)

    maybe_barrier()
    full_path = MODEL_DIR / "BPTI_1UAO_AA_NLL_T500_ddp.pic"
    if DDP["is_main"]:
        torch.save(generator.state_dict(), full_path)
        rank_print(f"saved: {ca_fixed_path}")
        rank_print(f"saved: {ca_torsion_path}")
        rank_print(f"saved: {nll_path}")
        rank_print(f"saved: {full_path}")
    maybe_barrier()

    if DDP["is_main"]:
        plot_report(ca_fixed_trainer, title_prefix="CA Fixed | ")
        plot_report(ca_torsion_trainer, title_prefix="CA Torsion | ")
        plot_report(nll_trainer, title_prefix="AA NLL | ")
        plot_report(full_nll_trainer, title_prefix="Full NLL | ")


if __name__ == "__main__":
    try:
        run()
    finally:
        cleanup_distributed()
