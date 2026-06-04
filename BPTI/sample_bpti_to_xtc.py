#!/usr/bin/env python
"""
Sample a trained BPTI Boltzmann generator and save samples as XTC.

/home/dell/software/miniforge3/envs/bgflow/bin/python experiment/BPTI/sample_bpti_to_xtc.py \
  --n-samples 1000 \
  --batch-size 100 \
  --device cuda:0 \
  --output experiment/BPTI/BPTI_BG_samples.xtc


"""

import argparse
import random
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from openmm import unit

import bgflow as bg
from bgflow import ANGLES, BONDS, FIXED, TORSIONS, InternalCoordinateMarginals, ShapeDictionary
from bgmol import bond_constraints
from bgmol.datasets.base import DataSet
from bgmol.systems.base import OpenMMSystem
from bgmol.util.importing import import_openmm
from bgmol.zmatrix import ZMatrixFactory, build_fake_topology


PROJECT_DIR = Path(__file__).resolve().parent
TOP_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_nowat_align.pdb"
TRAJ_T500_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_traj_nowat_align.dcd"
DEFAULT_CHECKPOINT = PROJECT_DIR / "models" / "BPTI_1UAO_AA_NLL_T500_ddp.pic"
DEFAULT_OUTPUT = PROJECT_DIR / "BPTI_BG_samples.xtc"

_, unit, app = import_openmm()


class BPTIImplicit(OpenMMSystem):
    def __init__(self, constraints=app.HBonds, hydrogen_mass=4.0 * unit.amu):
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


class BPTIImplicitThigh(DataSet):
    def __init__(self, root=PROJECT_DIR, download=False, read=False):
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_all_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        n_bad = torch.numel(tensor) - torch.isfinite(tensor).sum().item()
        raise RuntimeError(f"{name} contains {n_bad} non-finite values")


def check_coordinate_transform_is_finite(coordinate_transform, data, batchsize=64):
    with torch.no_grad():
        *outputs, dlogp = coordinate_transform(data[:batchsize])
    for name, tensor in zip(("bonds", "angles", "torsions", "fixed"), outputs):
        assert_all_finite(f"coordinate_transform {name}", tensor)
    assert_all_finite("coordinate_transform dlogp", dlogp)


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
        if ca_idx is not None:
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


def build_generator(dataset, device, dtype=torch.float32):
    ctx = {"dtype": dtype, "device": device}
    system = dataset.system
    coordinates = dataset.xyz

    generator = torch.Generator()
    generator.manual_seed(20260529)
    indices = torch.randperm(len(coordinates), generator=generator)
    training = torch.tensor(coordinates[indices[: int(len(coordinates) * 0.8)]], dtype=dtype, device=device)

    c_alpha = system.mdtraj_topology.select("name CA")
    zfactory = ZMatrixFactory(system.mdtraj_topology, cartesian=c_alpha)
    z_matrix, fixed_atoms = zfactory.build_naive()

    coordinate_transform = bg.RelativeInternalCoordinateTransformation(
        z_matrix=z_matrix,
        fixed_atoms=fixed_atoms,
        normalize_angles=True,
    ).to(**ctx)
    check_coordinate_transform_is_finite(coordinate_transform, training)

    shape_info = bg.ShapeDictionary.from_coordinate_transform(
        coordinate_transform,
        n_constraints=system.system.getNumConstraints(),
    )

    ca_fixed_atoms = select_ca_fixed_atoms_from_secondary_structure(dataset.trajectory, c_alpha)
    dssp = md.compute_dssp(dataset.trajectory[0], simplified=True)[0]
    print("Selected CA fixed atoms:")
    for ca_idx in ca_fixed_atoms:
        atom = dataset.trajectory.topology.atom(int(c_alpha[ca_idx]))
        residue = atom.residue
        print(f"  ca_idx={ca_idx}, atom={int(c_alpha[ca_idx])}, residue={residue}, ss={dssp[residue.index]}")

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

    builder = bg.BoltzmannGeneratorBuilder(shape_info, target=None, **ctx)
    builder.add_layer(ca_ic_generator._flow, [CA_BONDS, CA_ANGLES, CA_TORSIONS, CA_FIXED])

    from bgflow.factory.GNN_factory import allegro_hparams as hparams, make_allegro_config_dict as make_config_dict
    from bgflow.nn.periodic import WrapDistances

    distances_net = WrapDistances(torch.nn.Identity())
    fixed = coordinate_transform.forward(training[:1000])[3]
    rbf_distances = distances_net(fixed.detach()).flatten().cpu()
    rbf_distances = rbf_distances[rbf_distances <= hparams["r_max"]]

    avg_num_neighbors = ((len(rbf_distances) / (1000 * fixed.shape[1] / 3))) * 2
    layers = make_config_dict(**hparams)
    layers["radial_basis"][1]["basis_kwargs"]["data"] = rbf_distances
    layers["allegro"][1]["avg_num_neighbors"] = avg_num_neighbors

    from nequip.nn import SequentialGraphNetwork
    from bgflow.factory.GNN_factory import NequipWrapper

    gnn = SequentialGraphNetwork.from_parameters(shared_params=None, layers=layers)
    gnn_feature_extractor = NequipWrapper(gnn, cutoff=hparams["r_max"])

    builder.default_conditioner_type = "GNN"
    builder.default_conditioner_kwargs = {
        "r_max": hparams["r_max"],
        "GNN": gnn_feature_extractor,
        "use_checkpointing": False,
        "GNN_output_dim": hparams["GNN_feature_dim"] * len(c_alpha),
        "attention_units": len(c_alpha),
        "attention_level": "MHA",
    }

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
        torch.tensor(dataset.xyz[:, c_alpha, :], **ctx).reshape(-1, len(c_alpha) * 3),
        ca_crd_transform,
        bonds=CA_BONDS,
        angles=None,
        torsions=None,
    )
    builder.add_map_to_ic_domains(ca_cdfs, return_layers=True)
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
        builder.add_condition(t1, on=(FIXED, t2), hidden=(512, 512), param_groups=("aa", "torsions"))
        builder.add_condition(t2, on=(t1, FIXED), hidden=(512, 512), param_groups=("aa", "torsions"))
    builder.add_merge(
        (t1, t2),
        to=TORSIONS,
        sizes_or_indices=[np.arange(0, n_torsions, 2), np.arange(1, n_torsions, 2)],
    )

    for _ in range(2):
        builder.add_condition(BONDS, on=(ANGLES, FIXED, TORSIONS), hidden=(512, 512), param_groups=("aa",))
        builder.add_condition(ANGLES, on=(BONDS, TORSIONS, FIXED), hidden=(512, 512), param_groups=("aa",))

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
        torch.tensor(dataset.xyz[::2000], **ctx).reshape(len(dataset.xyz[::2000]), -1),
        coordinate_transform,
        constrained_bond_indices=bond_constraints(system.system, coordinate_transform)[0],
        torsions=None,
    )
    builder.add_map_to_ic_domains(cdfs, return_layers=True)
    builder.add_merge_constraints(*bond_constraints(system.system, coordinate_transform))
    builder.add_map_to_cartesian(coordinate_transform)

    return builder.build_generator(zero_parameters=False, check_target=False).to(**ctx)


def load_checkpoint(generator, checkpoint, device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location=device)
    generator.load_state_dict(state_dict, strict=True)


def sample_batches(generator, n_samples, batch_size, temperature, device):
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    samples = []
    remaining = n_samples
    with torch.no_grad():
        while remaining > 0:
            current = min(batch_size, remaining)
            batch = generator.sample(current, temperature=temperature)
            if isinstance(batch, tuple):
                if len(batch) != 1:
                    raise RuntimeError(f"Expected one sampled tensor, got {len(batch)}")
                batch = batch[0]
            batch = batch.detach().to(device="cpu")
            assert_all_finite("sample batch", batch)
            samples.append(batch)
            remaining -= current
    return torch.cat(samples, dim=0)


def tensor_to_trajectory(samples, topology):
    xyz = samples.reshape(samples.shape[0], topology.n_atoms, 3).numpy()
    return md.Trajectory(xyz=xyz, topology=topology)


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    dataset = BPTIImplicitThigh(read=True)
    generator = build_generator(dataset, device=device)
    load_checkpoint(generator, args.checkpoint, device)
    generator.eval()

    samples = sample_batches(generator, args.n_samples, args.batch_size, args.temperature, device)
    trajectory = tensor_to_trajectory(samples, dataset.system.mdtraj_topology)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save_xtc(str(args.output))
    print(f"saved {len(trajectory)} samples to {args.output}")


if __name__ == "__main__":
    main()
