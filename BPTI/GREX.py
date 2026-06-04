#!/usr/bin/env python
# coding: utf-8
"""
BPTI GREX run using a trained BPTI-CF-bbp flow.

Example:
  conda activate bgflow
  python GREX.py \
    --checkpoint models/BPTI_CF_bbp_ddp.pth \
    --sample BPTI_BG_samples_optimized_rmsd_filtered.xtc\
    --device cuda:0 \
    --cuda-device-index 0

Resume GREX_BPTI.xtc to 200 ns total:
  python GREX.py \
    --mode grex \
    --reservoir-file GREX_BPTI_reservoir.pt \
    --resume-from GREX_BPTI.xtc \
    --output GREX_BPTI.xtc \
    --target-ns 200 \
    --device cuda:0 \
    --cuda-device-index 0
"""

import argparse
import importlib.util
import os
import random
import tempfile
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform
from openmm.app import ForceField, HBonds, Modeller, PDBFile, PME, Simulation
from openmm.unit import bar, dalton, femtosecond, kelvin, nanometer, picosecond

import bgflow as bg


PROJECT_DIR = Path(__file__).resolve().parent
FLOW_SCRIPT = PROJECT_DIR / "BPTI-CF-bbp.py"
TOP_FILE = PROJECT_DIR / "1187409s_pdb" / "5PTI.pdb"
SOLVATED_PDB = PROJECT_DIR / "md_T300_10us_explicit_5PTI" / "bpti_solvated_start.pdb"
DEFAULT_SAMPLE = PROJECT_DIR / "BPTI_BG_samples.xtc"
DEFAULT_CHECKPOINT = PROJECT_DIR / "models" / "BPTI_CF_bbp_ddp.pth"
DEFAULT_OUTPUT = PROJECT_DIR / "GREX_BPTI_3.xtc"
DEFAULT_REPORT = PROJECT_DIR / "GREX_BPTI_state.npz"
DEFAULT_RESERVOIR = PROJECT_DIR / "GREX_BPTI_reservoir.pt"

AMINO_ACID_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "CYX",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def parse_args():
    default_device = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reservoir", "grex", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--sample-top", type=Path, default=TOP_FILE)
    parser.add_argument("--solvated-pdb", type=Path, default=SOLVATED_PDB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reservoir-file", type=Path, default=DEFAULT_RESERVOIR)
    parser.add_argument("--reservoir-xtc", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Existing protein-only GREX XTC to continue from. The original file is not modified until merge completes.",
    )
    parser.add_argument(
        "--resume-segment-output",
        type=Path,
        default=None,
        help="Temporary/new segment XTC written during resume before merging with --resume-from.",
    )
    parser.add_argument(
        "--target-ns",
        type=float,
        default=None,
        help="Run only enough additional saved frames to reach this total GREX time in ns.",
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--cuda-device-index", default=os.environ.get("CUDA_DEVICE_INDEX"))
    parser.add_argument("--platform", default=os.environ.get("OPENMM_PLATFORM", "CUDA"))
    parser.add_argument("--conditioner", default=os.environ.get("CONDITIONER"))
    parser.add_argument("--openmm-workers", type=int, default=int(os.environ.get("OPENMM_WORKERS", "16")))
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("GREX_BATCH_SIZE", "512")))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--logw-mode",
        choices=("neg_energy2", "full", "energy_diff", "dlogp_minus_energy2"),
        default="neg_energy2",
    )
    parser.add_argument("--n-iter", type=int, default=10**5)
    parser.add_argument("--pace", type=int, default=500)
    parser.add_argument("--mc-stride", type=int, default=100)
    parser.add_argument("--trajectory-save-interval", type=int, default=100)
    parser.add_argument("--n-equil", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--padding-nm", type=float, default=1.0)
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--barostat", action="store_true")
    parser.add_argument("--minimize-initial", action="store_true")
    parser.add_argument("--logw2-threshold", type=float, default=0.0)
    parser.add_argument("--print-interval", type=int, default=100)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_flow_module():
    spec = importlib.util.spec_from_file_location("bpti_cf_bbp", FLOW_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import flow script: {FLOW_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_cuda_device_index(device):
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is not None:
        return str(device.index)
    return "0"


def make_platform(platform_name, device, cuda_device_index):
    if platform_name.lower() == "auto":
        return None, {}

    platform = Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name.upper() == "CUDA":
        properties["DeviceIndex"] = cuda_device_index or infer_cuda_device_index(device)
    return platform, properties


def get_protein_atoms(topology):
    atoms = [atom.index for atom in topology.atoms() if atom.residue.name in AMINO_ACID_RESIDUES]
    if not atoms:
        raise RuntimeError("No protein atoms were selected from the topology")
    return atoms


def make_explicit_system(modeller, args, freeze_atom_indices=()):
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=args.nonbonded_cutoff_nm * nanometer,
        constraints=HBonds,
    )
    # if args.barostat:
    #     system.addForce(MonteCarloBarostat(1 * bar, args.temperature * kelvin))
    for atom_idx in freeze_atom_indices:
        system.setParticleMass(int(atom_idx), 0.0 * dalton)
    return system


def make_simulation(modeller, args, platform, properties, freeze_atom_indices=(), minimize=False):
    system = make_explicit_system(modeller, args, freeze_atom_indices=freeze_atom_indices)
    integrator = LangevinMiddleIntegrator(
        args.temperature * kelvin,
        args.friction / picosecond,
        args.timestep_fs * femtosecond,
    )
    if platform is None:
        simulation = Simulation(modeller.topology, system, integrator)
    else:
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)
    if minimize:
        simulation.minimizeEnergy()
    return simulation


def make_solvated_modeller_from_protein(system_nowat, protein_positions, args):
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    if torch.is_tensor(protein_positions):
        positions = protein_positions.reshape(*system_nowat.positions.shape).detach().cpu().numpy()
    else:
        positions = np.asarray(protein_positions, dtype=np.float32).reshape(*system_nowat.positions.shape)
    modeller = Modeller(system_nowat._topology, positions * nanometer)
    modeller.addSolvent(forcefield, model="tip3p", padding=args.padding_nm * nanometer)
    return modeller


def load_samples(sample_path, top_path, n_atoms, max_samples=None):
    sample_path = Path(sample_path)
    if sample_path.suffix == ".npy":
        sample = np.load(sample_path)
    elif sample_path.suffix in {".xtc", ".dcd"}:
        sample = md.load(str(sample_path), top=str(top_path)).xyz
    else:
        raise ValueError(f"Unsupported sample format: {sample_path}")

    if sample.ndim == 2:
        sample = sample.reshape(-1, n_atoms, 3)
    elif sample.ndim == 3:
        sample = sample.reshape(-1, n_atoms, 3)
    else:
        raise ValueError(f"Expected sample shape (N, D) or (N, atoms, 3), got {sample.shape}")

    if sample.shape[1] != n_atoms:
        raise ValueError(f"Sample has {sample.shape[1]} atoms, expected {n_atoms}")
    if max_samples is not None:
        sample = sample[:max_samples]
    return sample.astype(np.float32, copy=False)


def load_checkpoint(checkpoint):
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint, map_location="cpu")


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def resolve_conditioner(args, checkpoint=None):
    requested = args.conditioner.lower() if args.conditioner else None
    checkpoint_conditioner = None
    if isinstance(checkpoint, dict):
        checkpoint_conditioner = checkpoint.get("config", {}).get("conditioner")
        if checkpoint_conditioner is not None:
            checkpoint_conditioner = str(checkpoint_conditioner).lower()

    if requested is None:
        return checkpoint_conditioner or "allegro"
    if checkpoint_conditioner is not None and requested != checkpoint_conditioner:
        raise ValueError(
            f"Checkpoint was saved with conditioner={checkpoint_conditioner!r}, "
            f"but --conditioner={requested!r}. Use the matching conditioner or omit --conditioner."
        )
    return requested


def load_bpti_cf_bbp(args, ctx):
    checkpoint = None
    if args.checkpoint.exists():
        checkpoint = load_checkpoint(args.checkpoint)
    elif not args.allow_untrained:
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Train BPTI-CF-bbp.py first or pass --allow-untrained for a smoke run."
        )

    cf = import_flow_module()
    cf.DDP = cf.init_distributed(required_world_size=1)
    cf.ctx = ctx
    cf.CONDITIONER = resolve_conditioner(args, checkpoint)
    cf.OPENMM_WORKERS = args.openmm_workers
    cf.set_seed(args.seed)

    dataset_high = cf.BPTIImplicitThigh(download=False, read=True)
    system_1 = dataset_high.system
    dataset_low = cf.BPTIImplicitTlow(download=False, read=False)
    system_2 = dataset_low.system

    high_energy = dataset_high.get_energy_model(n_workers=args.openmm_workers)
    low_energy = dataset_low.get_energy_model(n_workers=args.openmm_workers)

    target = low_energy
    if cf.CLIPPING:
        target = bg.GradientClippedEnergy(
            low_energy,
            bg.utils.ClipGradient(clip=cf.CLIPPING, norm_dim=3),
        ).to(**ctx)

    training_data = torch.as_tensor(dataset_high.xyz[: min(1000, len(dataset_high.xyz))], **ctx).reshape(-1, system_1.dim)
    bgmap = cf.build_bgmap(dataset_high, system_1, target, training_data)

    if checkpoint is not None:
        bgmap.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
        print(f"loaded checkpoint: {args.checkpoint}", flush=True)
    elif args.allow_untrained:
        print(f"[WARN] checkpoint not found, using untrained flow: {args.checkpoint}", flush=True)
    bgmap.eval()
    return bgmap, dataset_high, system_1, system_2, high_energy, low_energy


def make_low_target(low_energy, ctx):
    if getattr(bg, "GradientClippedEnergy", None) is None:
        return low_energy
    return bg.GradientClippedEnergy(
        low_energy,
        bg.utils.ClipGradient(clip=100.0, norm_dim=3),
    ).to(**ctx)


def close_energy_model(energy_model):
    delegate = getattr(energy_model, "delegate", energy_model)
    bridge = getattr(delegate, "bridge", None)
    if bridge is None:
        return
    wrapper = getattr(bridge, "context_wrapper", None)
    terminate = getattr(wrapper, "terminate", None)
    if terminate is not None:
        terminate()


def close_energy_models(*energy_models):
    for energy_model in energy_models:
        close_energy_model(energy_model)


def load_bpti_systems_for_grex(args, ctx):
    cf = import_flow_module()
    cf.DDP = cf.init_distributed(required_world_size=1)
    cf.ctx = ctx
    cf.OPENMM_WORKERS = args.openmm_workers
    cf.set_seed(args.seed)

    dataset_high = cf.BPTIImplicitThigh(download=False, read=False)
    system_1 = dataset_high.system
    dataset_low = cf.BPTIImplicitTlow(download=False, read=False)
    system_2 = dataset_low.system
    low_energy = dataset_low.get_energy_model(n_workers=args.openmm_workers)
    low_energy = make_low_target(low_energy, ctx)
    return system_1, system_2, low_energy


def reservoir_logw(mode, energy_1, dlogp_h, energy_2):
    if mode == "neg_energy2":
        return -energy_2
    if mode == "full":
        return energy_1 + dlogp_h - energy_2
    if mode == "energy_diff":
        return energy_1 - energy_2
    if mode == "dlogp_minus_energy2":
        return dlogp_h - energy_2
    raise ValueError(f"Unsupported logw mode: {mode}")


def precompute_reservoir(sample, bgmap, system_1, high_energy, low_energy, args, ctx):
    z_list = []
    dlogp_list = []
    logw_h_list = []

    with torch.no_grad():
        for i in range(0, len(sample), args.batch_size):
            batch = sample[i : i + args.batch_size]
            batch_tensor = torch.as_tensor(batch.reshape(len(batch), system_1.dim), **ctx)

            try:
                z_l, dlogp_h = bgmap(batch_tensor)
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"CUDA OOM while precomputing reservoir with batch_size={args.batch_size}. "
                    "Retry with a smaller --batch-size, e.g. 256 or 128."
                ) from exc
            energy_1 = high_energy.energy(batch_tensor)
            energy_2 = low_energy.energy(z_l)
            logw_h = reservoir_logw(args.logw_mode, energy_1, dlogp_h, energy_2)

            z_list.append(z_l.detach().cpu())
            dlogp_list.append(dlogp_h.detach().cpu())
            logw_h_list.append(logw_h.detach().cpu())
            print(f"reservoir... {min(i + len(batch), len(sample))}/{len(sample)}", end="\r", flush=True)

    print("", flush=True)
    z_l_all = torch.cat(z_list, dim=0)
    dlogp_all = torch.cat(dlogp_list, dim=0)
    logw_h_all = torch.cat(logw_h_list, dim=0)
    return z_l_all, dlogp_all, logw_h_all


def save_reservoir(path, z_l_all, dlogp_all, logw_h_all, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "z_l_all": z_l_all.detach().cpu(),
            "dlogp_all": dlogp_all.detach().cpu(),
            "logw_h_all": logw_h_all.detach().cpu(),
            "metadata": {
                "sample": str(args.sample),
                "sample_top": str(args.sample_top),
                "checkpoint": str(args.checkpoint),
                "logw_mode": args.logw_mode,
                "max_samples": args.max_samples,
            },
        },
        path,
    )


def reservoir_xtc_path(args):
    if args.reservoir_xtc is not None:
        return args.reservoir_xtc
    return args.reservoir_file.with_suffix(".xtc")


def save_reservoir_xtc(path, z_l_all, system):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = z_l_all.detach().cpu().reshape(-1, *system.positions.shape).numpy()
    trajectory = md.Trajectory(xyz=xyz, topology=system.mdtraj_topology)
    trajectory.save_xtc(str(path))


def saved_frame_dt_ns(pace, timestep_fs):
    return float(pace) * float(timestep_fs) / 1_000_000.0


def remaining_iterations_for_target_ns(existing_frames, target_ns, pace, timestep_fs):
    if target_ns is None:
        raise ValueError("target_ns must not be None")
    if target_ns < 0:
        raise ValueError(f"target_ns must be non-negative, got {target_ns}")
    frame_dt = saved_frame_dt_ns(pace, timestep_fs)
    if frame_dt <= 0:
        raise ValueError(f"Saved frame dt must be positive, got {frame_dt}")
    target_frames = int(np.ceil(float(target_ns) / frame_dt))
    return max(0, target_frames - int(existing_frames))


def default_resume_segment_output(output):
    output = Path(output)
    return output.with_name(f"{output.stem}.segment{output.suffix}")


def count_xtc_frames(path, topology):
    total = 0
    for chunk in md.iterload(str(path), top=topology, chunk=1000):
        total += chunk.n_frames
    return total


def load_last_xtc_frame(path, topology):
    last = None
    for chunk in md.iterload(str(path), top=topology, chunk=1000):
        if chunk.n_frames:
            last = chunk[-1]
    if last is None:
        raise ValueError(f"Resume trajectory has no frames: {path}")
    return last.xyz[0].astype(np.float32, copy=False)


def merge_xtc_files(base_path, segment_path, output_path, topology):
    base_path = Path(base_path)
    segment_path = Path(segment_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    same_output = output_path.resolve() == base_path.resolve()
    if same_output:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.merge.",
            suffix=output_path.suffix,
            dir=output_path.parent,
            delete=False,
        )
        merged_path = Path(handle.name)
        handle.close()
    else:
        merged_path = output_path

    try:
        with md.formats.XTCTrajectoryFile(str(merged_path), "w") as xtc_file:
            offset = 0
            for source in (base_path, segment_path):
                for chunk in md.iterload(str(source), top=topology, chunk=1000):
                    step = np.arange(offset, offset + chunk.n_frames, dtype=np.int32)
                    xtc_file.write(chunk.xyz, step=step)
                    offset += chunk.n_frames
            xtc_file.flush()
        if same_output:
            os.replace(merged_path, output_path)
    except Exception:
        if same_output and merged_path.exists():
            merged_path.unlink()
        raise


def resolve_resume_run_lengths(args, topology):
    if args.resume_from is None:
        return args.n_iter, None
    if not args.resume_from.exists():
        raise FileNotFoundError(f"Resume trajectory not found: {args.resume_from}")
    existing_frames = count_xtc_frames(args.resume_from, topology)
    if args.target_ns is None:
        return args.n_iter, existing_frames
    remaining = remaining_iterations_for_target_ns(
        existing_frames=existing_frames,
        target_ns=args.target_ns,
        pace=args.pace,
        timestep_fs=args.timestep_fs,
    )
    return remaining, existing_frames


def load_reservoir(path):
    if not path.exists():
        raise FileNotFoundError(f"Reservoir file not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    required = ("z_l_all", "dlogp_all", "logw_h_all")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Reservoir file is missing keys {missing}: {path}")
    return state["z_l_all"], state["dlogp_all"], state["logw_h_all"], state.get("metadata", {})


def choose_candidate(z_l_all, logw_h_all, logw_2, device):
    idx = torch.randint(0, len(z_l_all), (1,), device=device).item()
    logw_1 = logw_h_all.reshape(-1)[idx].to(device=device, non_blocking=True)
    delta_energy = -(logw_1 + logw_2.reshape(1)[0])
    return z_l_all[idx].to(device=device, non_blocking=True), logw_1, float(delta_energy.item())


def should_save_trajectory(step_index, n_iter, interval):
    if interval <= 0:
        return step_index == n_iter - 1
    return (step_index + 1) % interval == 0 or step_index == n_iter - 1


def flush_trajectory_chunk(xtc_file, data, start_frame, stop_frame):
    if stop_frame <= start_frame:
        return start_frame
    step = np.arange(start_frame, stop_frame, dtype=np.int32)
    xtc_file.write(data[start_frame:stop_frame], step=step)
    xtc_file.flush()
    return stop_frame


def run_grex(args, system_1, system_2, low_energy, z_l_all, logw_h_all, ctx):
    platform, properties = make_platform(args.platform, ctx["device"], args.cuda_device_index)
    system_nowat = system_2

    run_n_iter, existing_frames = resolve_resume_run_lengths(args, system_1.mdtraj_topology)
    resume_segment_output = None
    original_output = args.output
    if args.resume_from is not None:
        if run_n_iter <= 0:
            print(
                f"resume target already reached: {existing_frames} frames "
                f"({existing_frames * saved_frame_dt_ns(args.pace, args.timestep_fs):.3f} ns)",
                flush=True,
            )
            return
        resume_segment_output = args.resume_segment_output or default_resume_segment_output(args.output)
        args.output = resume_segment_output
        last_protein_positions = load_last_xtc_frame(args.resume_from, system_1.mdtraj_topology)
        modeller = make_solvated_modeller_from_protein(system_nowat, last_protein_positions, args)
        print(
            f"resuming from {args.resume_from}: existing_frames={existing_frames}, "
            f"additional_iterations={run_n_iter}, segment={resume_segment_output}",
            flush=True,
        )
    else:
        pdb = PDBFile(str(args.solvated_pdb))
        modeller = Modeller(pdb.topology, pdb.positions)

    simulation = make_simulation(
        modeller,
        args,
        platform,
        properties,
        freeze_atom_indices=(),
        minimize=args.minimize_initial,
    )

    protein_atoms = get_protein_atoms(simulation.topology)
    if len(protein_atoms) != system_1.positions.shape[0]:
        raise RuntimeError(
            f"Selected {len(protein_atoms)} protein atoms from explicit topology, "
            f"expected {system_1.positions.shape[0]}"
        )
    print(f"protein_atoms: {len(protein_atoms)}", flush=True)

    if args.n_equil > 0:
        print("-------------equilibrating-------------", flush=True)
        simulation.step(args.n_equil * args.pace)

    print("-------------MD-------------", flush=True)
    n_attempts = (run_n_iter - 1) // args.mc_stride + 1
    data = np.zeros((run_n_iter, *system_1.positions.shape), dtype=np.float32)
    mapped_high = np.full((n_attempts, *system_1.positions.shape), np.nan, dtype=np.float32)
    logw_high = np.full(n_attempts, np.nan, dtype=np.float32)
    delta_high = np.full(n_attempts, np.nan, dtype=np.float32)
    is_swapped = np.full(run_n_iter, False, dtype=bool)

    attempt_idx = 0
    last_saved_frame = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with md.formats.XTCTrajectoryFile(str(args.output), "w") as xtc_file:
        for n in range(run_n_iter):
            simulation.step(args.pace)
            state = simulation.context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            protein_positions = positions[protein_atoms]
            data[n] = protein_positions

            if (n + 1) % args.mc_stride == 0:
                with torch.no_grad():
                    md_l = torch.as_tensor(data[n], **ctx).view(1, -1)
                    logw_2 = low_energy.energy(md_l)
                    map_l, logw_1, delta_energy = choose_candidate(z_l_all, logw_h_all, logw_2, ctx["device"])
                    # print(logw_1, logw_2, delta_energy)

                mapped_high[attempt_idx] = map_l.view(*system_1.positions.shape).detach().cpu().numpy()
                logw_high[attempt_idx] = float(logw_1.item())
                delta_high[attempt_idx] = delta_energy

                accept = delta_energy <= 0.0 or np.random.rand() < np.exp(-delta_energy)
                accept = accept and float(logw_2.item()) < args.logw2_threshold
                if accept:
                    is_swapped[n] = True
                    protein_pos = map_l.view(*system_1.positions.shape)

                    freeze_indices = range(system_1.positions.shape[0])
                    solvated = make_solvated_modeller_from_protein(system_nowat, protein_pos, args)
                    water_opt_sim = make_simulation(
                        solvated,
                        args,
                        platform,
                        properties,
                        freeze_atom_indices=freeze_indices,
                        minimize=True,
                    )
                    optimized_positions = water_opt_sim.context.getState(getPositions=True).getPositions(asNumpy=True)

                    solvated = make_solvated_modeller_from_protein(system_nowat, protein_pos, args)
                    simulation = make_simulation(
                        solvated,
                        args,
                        platform,
                        properties,
                        freeze_atom_indices=(),
                        minimize=False,
                    )
                    simulation.context.setPositions(optimized_positions)
                    protein_atoms = get_protein_atoms(simulation.topology)

                attempt_idx += 1

            if (n + 1) % args.print_interval == 0 or n == run_n_iter - 1:
                attempts = max(1, attempt_idx)
                print(
                    f"sampling... {(n + 1) / run_n_iter:.1%}, "
                    f"accepted: {is_swapped.sum():g} of {attempts} ({is_swapped.sum() / attempts:.2%})",
                    end="\r",
                    flush=True,
                )

            if should_save_trajectory(n, run_n_iter, args.trajectory_save_interval):
                last_saved_frame = flush_trajectory_chunk(xtc_file, data, last_saved_frame, n + 1)

    print("", flush=True)

    if args.resume_from is not None:
        merge_xtc_files(args.resume_from, resume_segment_output, original_output, system_1.mdtraj_topology)
        print(f"merged resume trajectory: {original_output}", flush=True)
        args.output = original_output

    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.report,
        mapped_high=mapped_high,
        logw_high=logw_high,
        delta_high=delta_high,
        is_swapped=is_swapped,
        output=str(args.output),
        sample=str(args.sample),
        checkpoint=str(args.checkpoint),
        resume_from=str(args.resume_from) if args.resume_from is not None else "",
        resume_segment_output=str(resume_segment_output) if resume_segment_output is not None else "",
        existing_frames=-1 if existing_frames is None else existing_frames,
        n_iter=run_n_iter,
        frame_dt_ns=saved_frame_dt_ns(args.pace, args.timestep_fs),
    )
    print(f"saved trajectory: {args.output}", flush=True)
    print(f"saved report: {args.report}", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    ctx = {"dtype": torch.float32, "device": device}

    if args.mode in {"reservoir", "both"}:
        bgmap, dataset_high, system_1, system_2, high_energy, low_energy = load_bpti_cf_bbp(args, ctx)
        try:
            sample = load_samples(args.sample, args.sample_top, system_1.positions.shape[0], max_samples=args.max_samples)
            print(f"sample shape: {sample.shape}", flush=True)
            z_l_all, dlogp_all, logw_h_all = precompute_reservoir(
                sample,
                bgmap,
                system_1,
                high_energy,
                low_energy,
                args,
                ctx,
            )
            save_reservoir(args.reservoir_file, z_l_all, dlogp_all, logw_h_all, args)
            print(f"saved reservoir: {args.reservoir_file}", flush=True)
            reservoir_xtc = reservoir_xtc_path(args)
            save_reservoir_xtc(reservoir_xtc, z_l_all, system_2)
            print(f"saved reservoir xtc: {reservoir_xtc}", flush=True)
            close_energy_model(high_energy)
            high_energy = None

            if args.mode == "reservoir":
                return

            print(
                f"reservoir tensors: z_l={tuple(z_l_all.shape)}, "
                f"dlogp={tuple(dlogp_all.shape)}, logw={tuple(logw_h_all.shape)}",
                flush=True,
            )
            run_grex(args, system_1, system_2, low_energy, z_l_all, logw_h_all, ctx)
        finally:
            close_energy_models(high_energy, low_energy)
    else:
        system_1, system_2, low_energy = load_bpti_systems_for_grex(args, ctx)
        try:
            z_l_all, dlogp_all, logw_h_all, metadata = load_reservoir(args.reservoir_file)
            print(f"loaded reservoir: {args.reservoir_file}", flush=True)
            if metadata:
                print(f"reservoir metadata: {metadata}", flush=True)

            print(
                f"reservoir tensors: z_l={tuple(z_l_all.shape)}, "
                f"dlogp={tuple(dlogp_all.shape)}, logw={tuple(logw_h_all.shape)}",
                flush=True,
            )
            run_grex(args, system_1, system_2, low_energy, z_l_all, logw_h_all, ctx)
        finally:
            close_energy_model(low_energy)


if __name__ == "__main__":
    main()
