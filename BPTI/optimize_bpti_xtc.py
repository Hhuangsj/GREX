#!/usr/bin/env python
"""
Optimize sampled BPTI structures with OpenMM and keep only accepted frames.

Example:
  python optimize_bpti_xtc.py \
    --input BPTI_BG_samples.xtc \
    --output BPTI_BG_samples_optimized.xtc \
    --energy-threshold -100 \
    --platform CUDA \
    --devices 0,1,2,3,4,5 \
    --progress-interval 10
"""

import argparse
import os
from multiprocessing import Pool, Process, Queue
from pathlib import Path

import mdtraj as md
import numpy as np
from openmm import LangevinMiddleIntegrator, Platform, unit
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation


PROJECT_DIR = Path(__file__).resolve().parent
TOP_FILE = PROJECT_DIR / "md_T500_50ns_explicit_5PTI" / "bpti_T500_nowat_align.pdb"
DEFAULT_INPUT = PROJECT_DIR / "BPTI_BG_samples.xtc"
DEFAULT_OUTPUT = PROJECT_DIR / "BPTI_BG_samples_optimized.xtc"
DEFAULT_RMSD_FILTERED_OUTPUT = PROJECT_DIR / "BPTI_BG_samples_optimized_rmsd_filtered.xtc"

_WORKER_SYSTEM = None
_WORKER_TOPOLOGY = None
_WORKER_TEMPERATURE = None
_WORKER_STEPS = None
_WORKER_MINIMIZE = None
_WORKER_THRESHOLD = None
_WORKER_PLATFORM = None
_WORKER_DEVICE = None
_WORKER_PRECISION = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--topology", type=Path, default=TOP_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indices-output", type=Path, default=None)
    parser.add_argument("--rmsd-only", action="store_true", help="Only filter an already optimized trajectory by RMSD.")
    parser.add_argument("--rmsd-reference", type=Path, default=TOP_FILE)
    parser.add_argument("--rmsd-selection", default="name CA")
    parser.add_argument("--rmsd-threshold", type=float, default=0.3, help="Discard frames with RMSD above this value in nm.")
    parser.add_argument("--rmsd-output", type=Path, default=None)
    parser.add_argument("--energy-threshold", type=float, default=-100.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--relax-steps", type=int, default=1000)
    parser.add_argument("--no-minimize", action="store_true")
    parser.add_argument("--platform", choices=("CUDA", "CPU"), default="CUDA")
    parser.add_argument("--devices", default="0,1,2,3,4,5")
    parser.add_argument("--cuda-precision", default="mixed")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 1) - 8))
    parser.add_argument("--progress-interval", type=int, default=1)
    return parser.parse_args()


def build_bpti_system(topology_file):
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    pdb = PDBFile(str(topology_file))
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        hydrogenMass=4.0 * unit.amu,
    )
    return system, pdb.topology


def parse_devices(devices):
    parsed = [device.strip() for device in devices.split(",") if device.strip()]
    if not parsed:
        raise ValueError("--devices must contain at least one CUDA device index")
    return parsed


def select_rmsd_atoms(topology, selection):
    atom_indices = topology.select(selection)
    if len(atom_indices) == 0:
        raise ValueError(f"RMSD selection matched no atoms: {selection!r}")
    return atom_indices


def rmsd_filter(trajectory, reference, atom_indices, cutoff_nm):
    if cutoff_nm < 0:
        raise ValueError("--rmsd-threshold must be >= 0")
    if reference.n_frames < 1:
        raise ValueError("RMSD reference has no frames")
    if trajectory.n_frames == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    target_xyz = trajectory.xyz[:, atom_indices].astype(np.float64, copy=False)
    reference_xyz = reference.xyz[0, atom_indices].astype(np.float64, copy=False)
    target_centered = target_xyz - target_xyz.mean(axis=1, keepdims=True)
    reference_centered = reference_xyz - reference_xyz.mean(axis=0, keepdims=True)

    covariance = np.einsum("fai,aj->fij", target_centered, reference_centered)
    left, _singular_values, right_t = np.linalg.svd(covariance)
    handedness = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", left, right_t)))
    handedness[handedness == 0.0] = 1.0
    correction = np.tile(np.eye(3), (trajectory.n_frames, 1, 1))
    correction[:, -1, -1] = handedness
    rotations = np.einsum("fij,fjk,fkl->fil", left, correction, right_t)
    aligned = np.einsum("fai,fij->faj", target_centered, rotations)
    squared_distances = ((aligned - reference_centered[None, :, :]) ** 2).sum(axis=2)
    rmsds = np.sqrt(squared_distances.mean(axis=1))
    kept_indices = np.flatnonzero(rmsds <= cutoff_nm + 1e-7)
    return rmsds, kept_indices


def save_filtered_trajectory(trajectory, kept_indices, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered = trajectory[kept_indices] if len(kept_indices) else trajectory[:0]
    filtered.save_xtc(str(output))
    return filtered


def save_rmsd_filter_outputs(trajectory, rmsds, kept_indices, output, indices_output=None):
    save_filtered_trajectory(trajectory, kept_indices, output)
    if indices_output is None:
        indices_output = output.with_suffix(".indices.txt")
    np.savetxt(indices_output, kept_indices.astype(np.int64), fmt="%d")
    np.savetxt(output.with_suffix(".rmsd.txt"), rmsds, fmt="%.8f")
    failed_count = trajectory.n_frames - len(kept_indices)
    print(f"RMSD filtering completed. Success: {len(kept_indices)}, Failed: {failed_count}")
    print(f"Saved RMSD-filtered frames to {output}")
    print(f"Saved accepted indices to {indices_output}")
    print(f"Saved all RMSD values to {output.with_suffix('.rmsd.txt')}")
    return output, indices_output


def run_rmsd_filter(args):
    if not args.input.exists():
        raise FileNotFoundError(f"Input trajectory not found: {args.input}")
    if not args.topology.exists():
        raise FileNotFoundError(f"Topology not found: {args.topology}")
    if not args.rmsd_reference.exists():
        raise FileNotFoundError(f"RMSD reference not found: {args.rmsd_reference}")

    trajectory = md.load(str(args.input), top=str(args.topology))
    reference = md.load(str(args.rmsd_reference), top=str(args.topology))
    atom_indices = select_rmsd_atoms(trajectory.topology, args.rmsd_selection)
    rmsds, kept_indices = rmsd_filter(trajectory, reference, atom_indices, args.rmsd_threshold)
    output = args.rmsd_output
    if output is None:
        output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_RMSD_FILTERED_OUTPUT

    print(f"Input frames: {trajectory.n_frames}")
    print(f"RMSD selection: {args.rmsd_selection} ({len(atom_indices)} atoms)")
    print(f"RMSD threshold: {args.rmsd_threshold} nm")
    print(
        "RMSD stats: "
        f"min={np.nanmin(rmsds):.4f} mean={np.nanmean(rmsds):.4f} max={np.nanmax(rmsds):.4f} nm"
    )
    save_rmsd_filter_outputs(trajectory, rmsds, kept_indices, output, args.indices_output)


def init_worker(topology_file, temperature, relax_steps, minimize, energy_threshold, platform_name, device, precision):
    global _WORKER_SYSTEM
    global _WORKER_TOPOLOGY
    global _WORKER_TEMPERATURE
    global _WORKER_STEPS
    global _WORKER_MINIMIZE
    global _WORKER_THRESHOLD
    global _WORKER_PLATFORM
    global _WORKER_DEVICE
    global _WORKER_PRECISION

    _WORKER_SYSTEM, _WORKER_TOPOLOGY = build_bpti_system(topology_file)
    _WORKER_TEMPERATURE = temperature
    _WORKER_STEPS = relax_steps
    _WORKER_MINIMIZE = minimize
    _WORKER_THRESHOLD = energy_threshold
    _WORKER_PLATFORM = platform_name
    _WORKER_DEVICE = device
    _WORKER_PRECISION = precision


def optimize_sample(args):
    index, xyz_nm = args
    try:
        platform = Platform.getPlatformByName(_WORKER_PLATFORM)
        if _WORKER_PLATFORM == "CUDA":
            properties = {"DeviceIndex": str(_WORKER_DEVICE), "CudaPrecision": _WORKER_PRECISION}
        else:
            properties = {"Threads": "1"}
        integrator = LangevinMiddleIntegrator(
            _WORKER_TEMPERATURE * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond,
        )
        simulation = Simulation(_WORKER_TOPOLOGY, _WORKER_SYSTEM, integrator, platform, properties)
        simulation.context.setPositions(xyz_nm * unit.nanometer)

        if _WORKER_MINIMIZE:
            simulation.minimizeEnergy()
        if _WORKER_STEPS > 0:
            simulation.step(_WORKER_STEPS)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        accepted = np.isfinite(energy) and energy < _WORKER_THRESHOLD
        return index, np.asarray(positions, dtype=np.float32), float(energy), bool(accepted)
    except Exception as exc:
        return index, None, np.nan, False


def gpu_worker(worker_id, device, tasks, queue, topology_file, temperature, relax_steps, minimize, energy_threshold, precision):
    init_worker(topology_file, temperature, relax_steps, minimize, energy_threshold, "CUDA", device, precision)
    for task in tasks:
        index, positions, energy, accepted = optimize_sample(task)
        queue.put((worker_id, index, positions, energy, accepted))
    queue.put((worker_id, None, None, np.nan, False))


def run_cpu(work, args, trajectory):
    results = []
    completed = 0
    success_count = 0
    failed_count = 0
    progress_interval = max(1, args.progress_interval)

    with Pool(
        processes=args.processes,
        initializer=init_worker,
        initargs=(
            str(args.topology),
            args.temperature,
            args.relax_steps,
            not args.no_minimize,
            args.energy_threshold,
            "CPU",
            None,
            args.cuda_precision,
        ),
    ) as pool:
        for result in pool.imap_unordered(optimize_sample, work):
            results.append(result)
            index, _positions, energy, accepted = result
            completed, success_count, failed_count = print_progress(
                completed,
                success_count,
                failed_count,
                trajectory.n_frames,
                index,
                energy,
                accepted,
                progress_interval,
            )
    return results


def run_cuda(work, args, trajectory):
    devices = parse_devices(args.devices)
    task_chunks = [work[i::len(devices)] for i in range(len(devices))]
    queue = Queue()
    processes = []
    for worker_id, (device, tasks) in enumerate(zip(devices, task_chunks)):
        process = Process(
            target=gpu_worker,
            args=(
                worker_id,
                device,
                tasks,
                queue,
                str(args.topology),
                args.temperature,
                args.relax_steps,
                not args.no_minimize,
                args.energy_threshold,
                args.cuda_precision,
            ),
        )
        process.start()
        processes.append(process)

    results = []
    completed = 0
    success_count = 0
    failed_count = 0
    finished_workers = 0
    progress_interval = max(1, args.progress_interval)

    while finished_workers < len(processes):
        worker_id, index, positions, energy, accepted = queue.get()
        if index is None:
            finished_workers += 1
            continue
        results.append((index, positions, energy, accepted))
        completed, success_count, failed_count = print_progress(
            completed,
            success_count,
            failed_count,
            trajectory.n_frames,
            index,
            energy,
            accepted,
            progress_interval,
            prefix=f"gpu={devices[worker_id]} ",
        )

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {process.exitcode}")
    return results


def print_progress(
    completed,
    success_count,
    failed_count,
    total,
    index,
    energy,
    accepted,
    progress_interval,
    prefix="",
):
    completed += 1
    if accepted:
        success_count += 1
    else:
        failed_count += 1

    if completed % progress_interval == 0 or completed == total:
        energy_text = "nan" if not np.isfinite(energy) else f"{energy:.3f}"
        print(
            f"[{completed}/{total}] {prefix}"
            f"success={success_count} failed={failed_count} "
            f"last_index={index} last_energy={energy_text} accepted={accepted}",
            flush=True,
        )
    return completed, success_count, failed_count


def main():
    args = parse_args()
    if args.rmsd_only:
        if args.input == DEFAULT_INPUT:
            args.input = DEFAULT_OUTPUT
        run_rmsd_filter(args)
        return

    if not args.input.exists():
        raise FileNotFoundError(f"Input trajectory not found: {args.input}")
    if not args.topology.exists():
        raise FileNotFoundError(f"Topology not found: {args.topology}")
    if not args.rmsd_reference.exists():
        raise FileNotFoundError(f"RMSD reference not found: {args.rmsd_reference}")

    trajectory = md.load(str(args.input), top=str(args.topology))
    reference = md.load(str(args.rmsd_reference), top=str(args.topology))
    atom_indices = select_rmsd_atoms(trajectory.topology, args.rmsd_selection)
    work = [(idx, trajectory.xyz[idx]) for idx in range(trajectory.n_frames)]

    if args.platform == "CUDA":
        devices = parse_devices(args.devices)
        print(f"Starting CUDA optimization on devices: {', '.join(devices)}")
    else:
        print(f"Starting CPU optimization with {args.processes} processes.")
    print(f"Input frames: {trajectory.n_frames}")
    print(f"Energy threshold: {args.energy_threshold} kJ/mol")

    if args.platform == "CUDA":
        results = run_cuda(work, args, trajectory)
    else:
        results = run_cpu(work, args, trajectory)

    kept_indices = []
    kept_xyz = []
    energies = np.full(trajectory.n_frames, np.nan, dtype=np.float64)
    for index, positions, energy, accepted in sorted(results, key=lambda item: item[0]):
        energies[index] = energy
        if accepted and positions is not None:
            kept_indices.append(index)
            kept_xyz.append(positions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if kept_xyz:
        optimized = md.Trajectory(xyz=np.asarray(kept_xyz), topology=trajectory.topology)
    else:
        optimized = md.Trajectory(xyz=np.empty((0, trajectory.n_atoms, 3)), topology=trajectory.topology)

    rmsds, rmsd_kept_indices = rmsd_filter(optimized, reference, atom_indices, args.rmsd_threshold)
    rmsd_kept_set = set(rmsd_kept_indices.tolist())
    final_indices = [index for local_idx, index in enumerate(kept_indices) if local_idx in rmsd_kept_set]
    final_xyz = optimized.xyz[rmsd_kept_indices] if len(rmsd_kept_indices) else np.empty((0, trajectory.n_atoms, 3))
    final_trajectory = md.Trajectory(xyz=final_xyz, topology=trajectory.topology)
    final_trajectory.save_xtc(str(args.output))

    indices_output = args.indices_output
    if indices_output is None:
        indices_output = args.output.with_suffix(".indices.txt")
    np.savetxt(indices_output, np.asarray(final_indices, dtype=np.int64), fmt="%d")
    np.savetxt(args.output.with_suffix(".energies.txt"), energies, fmt="%.8f")
    np.savetxt(args.output.with_suffix(".rmsd.txt"), rmsds, fmt="%.8f")

    failed_count = trajectory.n_frames - len(final_indices)
    rmsd_rejected = len(kept_indices) - len(final_indices)
    print(
        f"RMSD threshold: {args.rmsd_threshold} nm "
        f"using selection {args.rmsd_selection!r} ({len(atom_indices)} atoms)"
    )
    if len(rmsds):
        print(
            "Accepted-frame RMSD stats: "
            f"min={np.nanmin(rmsds):.4f} mean={np.nanmean(rmsds):.4f} max={np.nanmax(rmsds):.4f} nm"
        )
    print(
        f"Optimization completed. Success: {len(final_indices)}, Failed: {failed_count} "
        f"(RMSD rejected after energy filter: {rmsd_rejected})"
    )
    print(f"Saved optimized accepted frames to {args.output}")
    print(f"Saved accepted indices to {indices_output}")
    print(f"Saved all energies to {args.output.with_suffix('.energies.txt')}")
    print(f"Saved accepted-frame RMSD values to {args.output.with_suffix('.rmsd.txt')}")


if __name__ == "__main__":
    main()
