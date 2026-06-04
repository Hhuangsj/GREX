#!/usr/bin/env python
# coding: utf-8

import os
import random
import warnings
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from openmm import unit
from torch.utils.data import DataLoader, Dataset

import bgflow as bg
import bgmol
from bgflow import ANGLES, BONDS, FIXED, TORSIONS, InternalCoordinateMarginals
from bgmol.datasets.base import DataSet
from bgmol.systems.base import OpenMMSystem
from bgmol.util.importing import import_openmm
from bgmol.zmatrix import ZMatrixFactory

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as TorchDDP
except ImportError:
    dist = None
    TorchDDP = None


PROJECT_DIR = Path(__file__).resolve().parent
TOP_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_nowat_align.pdb"
TRAJ_T500_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_traj_nowat_align.dcd"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


DEFAULT_CHECKPOINT = MODEL_DIR / "BPTI_CF_bbp_ddp.pth"
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT)))
RESUME = env_flag("RESUME", False)
REQUIRED_WORLD_SIZE = int(os.environ.get("REQUIRED_WORLD_SIZE", "4"))
N_EPOCHS = int(os.environ.get("N_EPOCHS", "30"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", str(2**7)))
ENERGY_BATCH_SIZE = int(os.environ.get("ENERGY_BATCH_SIZE", str(2**7)))
OPENMM_WORKERS = int(os.environ.get("OPENMM_WORKERS", "8"))
OPENMM_PLATFORM = os.environ.get("OPENMM_PLATFORM", "CUDA" if torch.cuda.is_available() else "CPU")
OPENMM_PRECISION = os.environ.get("OPENMM_PRECISION", "mixed")
CUDA_DEVICE_INDEX = os.environ.get("CUDA_DEVICE_INDEX")
LR = float(os.environ.get("LR", "1e-4"))
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", "0.5"))
CLIPPING = float(os.environ.get("CLIPPING", "100"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
LAMBDA_ENERGY = float(os.environ.get("LAMBDA_ENERGY", "0.0"))
LAMBDA_RMSD = float(os.environ.get("LAMBDA_RMSD", "0.0"))
CONDITIONER = os.environ.get("CONDITIONER", "dense").lower()

DDP = None
ctx = None

_, unit, app = import_openmm()


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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if required_world_size is not None and required_world_size > 1:
            print(
                f"[WARN] Running single process (WORLD_SIZE=1). "
                f"For multi-GPU training use: torchrun --nproc_per_node={required_world_size} {Path(__file__).name}",
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
    if DDP is None or force or DDP["is_main"]:
        print(msg, flush=True)


def set_seed(base_seed=20260529):
    rank = 0 if DDP is None else DDP["rank"]
    seed = base_seed + rank
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
            f"Rank {rank} received empty shard (n={n}, world_size={world_size}); "
            "reduce world size or use more data"
        )
    return data[start:stop]


def local_batch_size(global_batch_size):
    if global_batch_size < 1:
        raise ValueError("global_batch_size must be >= 1")
    return max(1, global_batch_size // DDP["world_size"])


def distributed_mean(value):
    out = torch.tensor(float(value), device=ctx["device"], dtype=ctx["dtype"])
    if DDP["is_distributed"] and dist is not None and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out /= DDP["world_size"]
    return out.item()


def openmm_energy_kwargs():
    kwargs = {"platform_name": OPENMM_PLATFORM}
    if OPENMM_PLATFORM.upper() in {"CUDA", "OPENCL"}:
        device_index = CUDA_DEVICE_INDEX
        if device_index is None:
            device_index = str(DDP["local_rank"] if DDP is not None else 0)
        kwargs["platform_properties"] = {"DeviceIndex": str(device_index)}
        if OPENMM_PLATFORM.upper() == "CUDA" and OPENMM_PRECISION:
            kwargs["platform_properties"]["Precision"] = OPENMM_PRECISION
    else:
        kwargs["n_workers"] = OPENMM_WORKERS
    return kwargs


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_resume_checkpoint():
    if not RESUME:
        return None
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"RESUME=1 but checkpoint does not exist: {CHECKPOINT_PATH}")
    checkpoint = torch_load_checkpoint(CHECKPOINT_PATH, map_location=ctx["device"])
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing model_state_dict: {CHECKPOINT_PATH}")
    return checkpoint


def configure_resume_architecture(checkpoint):
    global CONDITIONER

    if checkpoint is None:
        return

    checkpoint_conditioner = checkpoint.get("config", {}).get("conditioner")
    if checkpoint_conditioner is None:
        return

    checkpoint_conditioner = str(checkpoint_conditioner).lower()
    if "CONDITIONER" not in os.environ:
        CONDITIONER = checkpoint_conditioner
        rank_print(f"resume: using conditioner={CONDITIONER!r} from checkpoint")
    elif CONDITIONER != checkpoint_conditioner:
        raise ValueError(
            f"Checkpoint was saved with conditioner={checkpoint_conditioner!r}, "
            f"but current CONDITIONER={CONDITIONER!r}. Use matching CONDITIONER or unset it."
        )


def completed_epochs_from_checkpoint(checkpoint):
    if checkpoint is None:
        return 0
    if "completed_epochs" in checkpoint:
        return int(checkpoint["completed_epochs"])
    if "epoch" in checkpoint:
        return int(checkpoint["epoch"])
    return max(len(checkpoint.get("train_reporter", [])), len(checkpoint.get("test_reporter", [])))


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
        if n_frames is not None:
            self.trajectory = self.trajectory[:n_frames]


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


class HighTempDataset(Dataset):
    def __init__(self, md_data, md_ener, mini_data, mini_ener, tensor_ctx):
        if not (len(md_data) == len(md_ener) == len(mini_data) == len(mini_ener)):
            raise ValueError("Data lengths do not match")

        self.md_data = torch.as_tensor(md_data, **tensor_ctx)
        self.md_ener = torch.as_tensor(md_ener, **tensor_ctx)
        self.mini_data = torch.as_tensor(mini_data, **tensor_ctx)
        self.mini_ener = torch.as_tensor(mini_ener, **tensor_ctx)

    def __len__(self):
        return len(self.md_data)

    def __getitem__(self, idx):
        return {
            "md_data": self.md_data[idx],
            "md_ener": self.md_ener[idx],
            "mini_data": self.mini_data[idx],
            "mini_ener": self.mini_ener[idx],
        }


def compute_rmsd(x, y, eps=1e-8):
    diff = x - y
    return torch.sqrt((diff**2).mean(dim=1) + eps)


def KLloss(
    prior_samples,
    prior_energies,
    min_data,
    min_ener,
    myflow,
    mytarget,
    lambda_energy=0.0,
    lambda_rmsd=0.0,
    test=False,
):
    mapped_samples, dlogp = myflow(prior_samples)
    mapped_energies = mytarget.energy(mapped_samples)

    kl_term = (mapped_energies - dlogp - prior_energies).mean()
    energy_diff = (mapped_energies - min_ener).mean()
    rmsd_diff = compute_rmsd(mapped_samples, min_data).mean()
    loss = kl_term + lambda_energy * energy_diff + lambda_rmsd * rmsd_diff

    if test and DDP["is_main"]:
        rank_print(f"KL term: {kl_term.item():.6f}")
        rank_print(f"Energy diff term: {(lambda_energy * energy_diff).item():.6f}")
        rank_print(f"RMSD term: {(lambda_rmsd * rmsd_diff).item():.6f}")
        rank_print(f"Total loss: {loss.item():.6f}")
        rank_print(f"dlogp mean: {dlogp.mean().item():.6f}, std: {dlogp.std().item():.6f}")

    return loss


def compute_energies(energy_model, data, batch_size):
    energies = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]
            energies.append(energy_model.energy(batch).detach())
    return torch.cat(energies, dim=0)


def make_dataloaders(train_tensors, test_tensors):
    train_set = HighTempDataset(*train_tensors, ctx)
    test_set = HighTempDataset(*test_tensors, ctx)

    train_generator = torch.Generator()
    train_generator.manual_seed(20260529 + DDP["rank"])

    train_loader = DataLoader(
        train_set,
        batch_size=local_batch_size(GLOBAL_BATCH_SIZE),
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=local_batch_size(GLOBAL_BATCH_SIZE),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def build_gnn_conditioner(builder, coordinate_transform, training_data, n_ca_atoms):
    from bgflow.factory.GNN_factory import make_allegro_config_dict as make_config_dict
    from bgflow.factory.GNN_factory import NequipWrapper
    from bgflow.nn.periodic import WrapDistances
    from nequip.nn import SequentialGraphNetwork

    hparams = {
        "r_max": 1.5,
        "num_types": 100,
        "num_basis": 32,
        "p": 6,
        "avg_num_neighbors": 9,
        "num_layers": 2,
        "env_embed_multiplicity": 32,
        "latent_dim": 32,
        "two_body_latent_intermediate_dims": [128, 128, 128],
        "nonscalars_include_parity": False,
        "irreps_edge_sh": "1x0e+1x1o+1x2e",
        "RBF_distance_offset": 1.0,
        "GNN_feature_dim": 16,
        "latent_resnet": True,
        "GNN_scope": "atomwise",
    }

    sample_size = min(1000, len(training_data))
    distances_net = WrapDistances(torch.nn.Identity())
    fixed = coordinate_transform.forward(training_data[:sample_size])[3]
    rbf_distances = distances_net(fixed.detach()).flatten().cpu()
    rbf_distances = rbf_distances[rbf_distances <= hparams["r_max"]]

    n_fixed_atoms = max(1.0, fixed.shape[1] / 3)
    if len(rbf_distances) > 0:
        avg_num_neighbors = (len(rbf_distances) / (sample_size * n_fixed_atoms)) * 2
    else:
        avg_num_neighbors = hparams["avg_num_neighbors"]

    layers = make_config_dict(**hparams)
    layers["radial_basis"][1]["basis_kwargs"]["data"] = rbf_distances
    layers["allegro"][1]["avg_num_neighbors"] = avg_num_neighbors

    gnn = SequentialGraphNetwork.from_parameters(shared_params=None, layers=layers)
    gnn_feature_extractor = NequipWrapper(gnn, cutoff=hparams["r_max"])

    builder.default_conditioner_type = "GNN"
    builder.default_conditioner_kwargs = {
        "r_max": hparams["r_max"],
        "GNN": gnn_feature_extractor,
        "use_checkpointing": False,
        "GNN_output_dim": hparams["GNN_feature_dim"] * n_ca_atoms,
        "attention_units": n_ca_atoms,
        "attention_level": "MHA",
    }


def build_bgmap(dataset_1, system_1, target, training_data):
    c_alpha = system_1.mdtraj_topology.select("name CA")
    zfactory = ZMatrixFactory(system_1.mdtraj_topology, cartesian=c_alpha)
    z_matrix, fixed_atoms = zfactory.build_naive()

    coordinate_transform = bg.RelativeInternalCoordinateTransformation(
        z_matrix=z_matrix,
        fixed_atoms=fixed_atoms,
        normalize_angles=True,
    ).to(**ctx)

    constrained_indices, constrained_lengths = bgmol.bond_constraints(system_1.system, coordinate_transform)
    shape_info = bg.ShapeDictionary.from_coordinate_transform(
        coordinate_transform,
        remove_origin_and_rotation=True,
        n_constraints=len(constrained_indices),
    )

    builder = bg.BoltzmannGeneratorBuilder(shape_info, target, **ctx)
    if CONDITIONER == "allegro":
        build_gnn_conditioner(builder, coordinate_transform, training_data, len(c_alpha))
    elif CONDITIONER == "dense":
        builder.default_conditioner_type = "dense"
        builder.default_conditioner_kwargs = {}
    else:
        raise ValueError(f"Unsupported CONDITIONER={CONDITIONER!r}; use 'dense' or 'allegro'")
    rank_print(f"conditioner: {CONDITIONER}")

    CA_REP = [FIXED]
    n_torsions = builder.current_dims[TORSIONS][-1]
    t1, t2 = builder.add_split(
        TORSIONS,
        into=["T1", "T2"],
        sizes_or_indices=[np.arange(0, n_torsions, 2), np.arange(1, n_torsions, 2)],
    )
    for _ in range(4):
        builder.add_condition(t1, on=(*CA_REP, t2), hidden=(128, 128), param_groups=("aa", "torsions"))
        builder.add_condition(t2, on=(t1, *CA_REP), hidden=(128, 128), param_groups=("aa", "torsions"))
    builder.add_merge(
        (t1, t2),
        to=TORSIONS,
        sizes_or_indices=[np.arange(0, n_torsions, 2), np.arange(1, n_torsions, 2)],
    )

    for _ in range(2):
        builder.add_condition(BONDS, on=(ANGLES, *CA_REP, TORSIONS), hidden=(128, 128), param_groups=("aa",))
        builder.add_condition(ANGLES, on=(BONDS, TORSIONS, *CA_REP), hidden=(128, 128), param_groups=("aa",))

    _ = InternalCoordinateMarginals(
        builder.current_dims,
        builder.ctx,
        bonds=BONDS,
        angles=ANGLES,
        torsions=None,
        fixed=None,
    )

    generator = builder.build_generator(zero_parameters=True).to(**ctx)

    edge_flow = []
    for _ in range(2):
        edge_builder = bg.BoltzmannGeneratorBuilder(shape_info, target, **ctx)
        edge_builder.add_merge_constraints(constrained_indices, constrained_lengths)
        edge_builder.add_map_to_cartesian(coordinate_transform)
        edge_flow.append(edge_builder.build_flow())

    bgmap = bg.SequentialFlow(
        [
            bg.InverseFlow(edge_flow[0]),
            generator._flow,
            edge_flow[1],
        ]
    ).to(**ctx)

    n_params = sum(p.numel() for p in bgmap.parameters() if p.requires_grad)
    rank_print(f"Number of trainable parameters: {n_params:_}")
    rank_print(f"dataset xyz shape: {dataset_1.xyz.shape}", force=True)
    rank_print(f"CA atoms: {len(c_alpha)}", force=True)

    return bgmap


def train_loop(dataloader, loss_fn, optimizer, params, reporter):
    epoch_losses = []

    for batch_idx, batch in enumerate(dataloader):
        loss = loss_fn(
            batch["md_data"],
            batch["md_ener"],
            batch["mini_data"],
            batch["mini_ener"],
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss on rank {DDP['rank']} batch {batch_idx}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP)
        optimizer.step()

        epoch_losses.append(loss.detach().item())

    local_avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    avg_loss = distributed_mean(local_avg)
    reporter.append(avg_loss)
    return avg_loss


def test_loss_once(dataloader, loss_fn, test_reporter=None):
    with torch.no_grad():
        for batch in dataloader:
            loss = loss_fn(
                batch["md_data"],
                batch["md_ener"],
                batch["mini_data"],
                batch["mini_ener"],
                test=True,
            )
            avg_loss = distributed_mean(loss.item())
            if test_reporter is not None:
                test_reporter.append(avg_loss)
            return avg_loss
    return float("nan")


def split_data(md_data):
    data_len = len(md_data)
    generator = torch.Generator()
    generator.manual_seed(20260529)
    indices = torch.randperm(data_len, generator=generator)

    len_training = int(data_len * TRAIN_FRACTION)
    if len_training <= 0 or len_training >= data_len:
        raise ValueError(f"TRAIN_FRACTION={TRAIN_FRACTION} leaves an empty train or test split")

    training_indices = indices[:len_training]
    testing_indices = indices[len_training:]

    training_data = md_data[training_indices]
    testing_data = md_data[testing_indices]

    if DDP["is_distributed"]:
        training_data = shard_tensor(training_data, DDP["world_size"], DDP["rank"])
        testing_data = shard_tensor(testing_data, DDP["world_size"], DDP["rank"])

    return (
        training_data.to(device=ctx["device"], non_blocking=True),
        testing_data.to(device=ctx["device"], non_blocking=True),
    )


def run():
    global DDP, ctx

    warnings.filterwarnings("ignore", category=UserWarning, message="InputOutsideDomain")

    DDP = init_distributed(required_world_size=REQUIRED_WORLD_SIZE)
    ctx = {"dtype": torch.float32, "device": DDP["device"]}
    set_seed(20260529)

    rank_print(f"[Rank {DDP['rank']}] device={ctx['device']} world_size={DDP['world_size']}", force=True)
    resume_checkpoint = load_resume_checkpoint()
    configure_resume_architecture(resume_checkpoint)

    dataset_1 = BPTIImplicitThigh(download=False, read=True)
    system_1 = dataset_1.system
    dataset_2 = BPTIImplicitTlow(download=False, read=False)

    energy_kwargs = openmm_energy_kwargs()
    rank_print(f"[Rank {DDP['rank']}] OpenMM energy kwargs: {energy_kwargs}", force=True)
    target_energy_1 = dataset_1.get_energy_model(**energy_kwargs)
    target_energy_2 = dataset_2.get_energy_model(**energy_kwargs)

    target = target_energy_2
    if CLIPPING:
        grad_clipping = bg.utils.ClipGradient(clip=CLIPPING, norm_dim=3)
        target = bg.GradientClippedEnergy(target_energy_2, grad_clipping).to(**ctx)
        rank_print(f"clipping atom forces at {CLIPPING:g}")

    md_data = torch.as_tensor(dataset_1.xyz, dtype=ctx["dtype"]).reshape(len(dataset_1.xyz), -1)
    training_data, testing_data = split_data(md_data)

    rank_print(f"[Rank {DDP['rank']}] train shard: {tuple(training_data.shape)}", force=True)
    rank_print(f"[Rank {DDP['rank']}] test shard: {tuple(testing_data.shape)}", force=True)

    training_ener = compute_energies(target_energy_1, training_data, ENERGY_BATCH_SIZE)
    testing_ener = compute_energies(target_energy_1, testing_data, ENERGY_BATCH_SIZE)

    train_tensors = (
        training_data,
        training_ener,
        training_data.clone(),
        training_ener.clone(),
    )
    test_tensors = (
        testing_data,
        testing_ener,
        testing_data.clone(),
        testing_ener.clone(),
    )
    train_loader, test_loader = make_dataloaders(train_tensors, test_tensors)

    bgmap = build_bgmap(dataset_1, system_1, target, training_data)
    params = [p for p in bgmap.parameters() if p.requires_grad]
    base_optimizer = torch.optim.Adam(params, lr=LR)
    optimizer = DistributedOptimizerWrapper(base_optimizer, params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, "min")

    completed_epochs = completed_epochs_from_checkpoint(resume_checkpoint)
    train_reporter = list(resume_checkpoint.get("train_reporter", [])) if resume_checkpoint is not None else []
    test_reporter = list(resume_checkpoint.get("test_reporter", [])) if resume_checkpoint is not None else []
    if resume_checkpoint is not None:
        bgmap.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            base_optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        rank_print(
            f"resume: loaded {CHECKPOINT_PATH}, completed_epochs={completed_epochs}, "
            f"training {N_EPOCHS} more epoch(s)"
        )

    for epoch in range(N_EPOCHS):
        bgmap.train(True)
        avg_train_loss = train_loop(
            train_loader,
            lambda prior_samples, prior_energies, min_data, min_ener: KLloss(
                prior_samples,
                prior_energies,
                min_data,
                min_ener,
                myflow=bgmap,
                mytarget=target,
                lambda_energy=LAMBDA_ENERGY,
                lambda_rmsd=LAMBDA_RMSD,
            ),
            optimizer,
            params,
            train_reporter,
        )

        bgmap.train(False)
        avg_test_loss = test_loss_once(
            test_loader,
            lambda prior_samples, prior_energies, min_data, min_ener, test=False: KLloss(
                prior_samples,
                prior_energies,
                min_data,
                min_ener,
                myflow=bgmap,
                mytarget=target,
                lambda_energy=LAMBDA_ENERGY,
                lambda_rmsd=LAMBDA_RMSD,
                test=test,
            ),
            test_reporter,
        )
        scheduler.step(avg_train_loss)

        current_lr = base_optimizer.param_groups[0]["lr"]
        global_epoch = completed_epochs + epoch + 1
        total_epochs = completed_epochs + N_EPOCHS
        rank_print(
            f"Epoch {global_epoch}/{total_epochs} "
            f"train={avg_train_loss:.6f} test={avg_test_loss:.6f} lr={current_lr:.3e}"
        )

    maybe_barrier()
    if DDP["is_main"]:
        torch.save(
            {
                "model_state_dict": bgmap.state_dict(),
                "optimizer_state_dict": base_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_reporter": train_reporter,
                "test_reporter": test_reporter,
                "completed_epochs": completed_epochs + N_EPOCHS,
                "config": {
                    "required_world_size": REQUIRED_WORLD_SIZE,
                    "n_epochs": completed_epochs + N_EPOCHS,
                    "resume": RESUME,
                    "checkpoint_path": str(CHECKPOINT_PATH),
                    "global_batch_size": GLOBAL_BATCH_SIZE,
                    "lr": LR,
                    "train_fraction": TRAIN_FRACTION,
                    "lambda_energy": LAMBDA_ENERGY,
                    "lambda_rmsd": LAMBDA_RMSD,
                    "conditioner": CONDITIONER,
                    "clipping": CLIPPING,
                    "grad_clip": GRAD_CLIP,
                },
            },
            CHECKPOINT_PATH,
        )
        rank_print(f"saved: {CHECKPOINT_PATH}")
    maybe_barrier()


if __name__ == "__main__":
    try:
        run()
    finally:
        cleanup_distributed()
