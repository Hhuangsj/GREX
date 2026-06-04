#!/usr/bin/env python
# coding: utf-8

# ## BPTI 300K 显式水 OpenMM 模拟（10 us）
# 
# 使用 PDB：`/home/dell/workstations/hsj_bgflow/experiment/BPTI/bgmol/data/bpti_top.pdb`。
# 这一段会生成 300 K 显式水轨迹，并把输出写到 `experiment/BPTI/md_T300_10us_explicit/`。
# 

from pathlib import Path
import time
import numpy as np

try:
    import openmm as mm
    from openmm import app, unit
except ImportError:
    from simtk import openmm as mm
    from simtk.openmm import app, unit

pdb_path = Path('/home/dell/workstations/hsj_bgflow/experiment/BPTI/1187409s_pdb/5PTI.pdb')
out_dir = Path('/home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit_5PTI')
out_dir.mkdir(parents=True, exist_ok=True)

T_init = 300.0
T_target = 300.0
pressure_atm = 1.0

dt_fs = 4.0
friction_ps = 1.0
padding_nm = 1.0
ionic_strength_M = 0.10

nvt_stage_steps = 2500      # 10 ps / stage
nvt_stages = 5              # 300 K -> 300 K
npt_equil_steps = 25000     # 100 ps
prod_steps = 2500000000     # 10 us at 4 fs
report_interval = 25000     # 100000 frames over 10 us

print(f'Input PDB: {pdb_path}')
print(f'Output dir: {out_dir}')


# In[10]:


pdb = app.PDBFile(str(pdb_path))
forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield, pH=7.0)
modeller.addSolvent(
    forcefield,
    model='tip3p',
    padding=padding_nm * unit.nanometer,
    ionicStrength=ionic_strength_M * unit.molar,
)

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
    hydrogenMass=4.0 * unit.amu,
    ewaldErrorTolerance=1e-4,
)

# barostat = mm.MonteCarloBarostat(pressure_atm * unit.atmosphere, T_target * unit.kelvin, 25)
# system.addForce(barostat)

solvated_pdb = out_dir / 'bpti_solvated_start.pdb'
with open(solvated_pdb, 'w') as f:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

print('Atoms in unsolvated PDB:', sum(1 for _ in pdb.topology.atoms()))
print('Atoms after solvation:', sum(1 for _ in modeller.topology.atoms()))
print(f'Saved solvated PDB: {solvated_pdb}')


# In[11]:


def choose_platform():
    for name, props in [
        ('CUDA', {'Precision': 'mixed', 'CudaDeviceIndex': '5', 'UseBlockingSync': 'false'}),
        ('OpenCL', {'Precision': 'mixed'}),
        ('CPU', {}),
        ('Reference', {}),
    ]:
        try:
            return mm.Platform.getPlatformByName(name), props
        except Exception:
            continue
    raise RuntimeError('No usable OpenMM platform found.')

if hasattr(mm, 'LangevinMiddleIntegrator'):
    integrator = mm.LangevinMiddleIntegrator(
        T_init * unit.kelvin,
        friction_ps / unit.picosecond,
        dt_fs * unit.femtosecond,
    )
else:
    integrator = mm.LangevinIntegrator(
        T_init * unit.kelvin,
        friction_ps / unit.picosecond,
        dt_fs * unit.femtosecond,
    )

platform, properties = choose_platform()
simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)
simulation.context.setPositions(modeller.positions)

print('OpenMM platform:', platform.getName())
if properties:
    print('Platform properties:', properties)


# In[12]:


log_csv = out_dir / "bpti_T300_state.csv"
traj_dcd = out_dir / "bpti_T300_traj.dcd"
chk_file = out_dir / "bpti_T300.chk"
final_pdb = out_dir / "bpti_T300_final.pdb"

# ===== 续跑判断 =====
resumed = chk_file.exists()
if resumed:
    print(f"Found checkpoint, resuming from: {chk_file}")
    simulation.loadCheckpoint(str(chk_file))
else:
    print("No checkpoint found, starting fresh.")

start = time.time()

# ===== 首次运行才做最小化和升温；续跑直接生产 =====
if not resumed:
    print("Minimization...")
    simulation.minimizeEnergy(maxIterations=1000)

    print(f"NVT heating: {T_init:.1f} K -> {T_target:.1f} K")
    temp_schedule = np.linspace(T_init, T_target, nvt_stages)
    for t in temp_schedule:
        integrator.setTemperature(float(t) * unit.kelvin)
        # 如你使用 barostat 且需要同步温度参数，可在这里 setParameter
        simulation.context.setVelocitiesToTemperature(float(t) * unit.kelvin)
        simulation.step(nvt_stage_steps)
else:
    print(f"Resumed. Continue production at {T_target:.1f} K ...")

# ===== Reporter：生产阶段才挂载；日志可追加；DCD 续跑时写新分段文件 =====
log_fh = open(log_csv, "a" if resumed else "w", buffering=1)

simulation.reporters.append(
    app.StateDataReporter(
        log_fh,
        report_interval,
        step=True,
        time=True,
        temperature=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        density=True,
        volume=True,
        speed=True,
        separator=",",
    )
)

if resumed:
    traj_dcd_run = out_dir / "bpti_T300_traj_resume.dcd"
    print(f"Resume mode: write continued trajectory to: {traj_dcd_run}")
else:
    traj_dcd_run = traj_dcd

simulation.reporters.append(app.DCDReporter(str(traj_dcd_run), report_interval))
simulation.reporters.append(app.CheckpointReporter(str(chk_file), report_interval))

print(f"Production at {T_target:.1f} K ...")
simulation.step(prod_steps)

state = simulation.context.getState(getPositions=True)
with open(final_pdb, "w") as f:
    app.PDBFile.writeFile(modeller.topology, state.getPositions(), f)

elapsed = time.time() - start
total_steps = nvt_stage_steps * nvt_stages + npt_equil_steps + prod_steps
total_ps = total_steps * dt_fs / 1000.0

print(f"Done. Simulated {total_ps:.1f} ps in {elapsed/60:.1f} min")
print(f"Trajectory: {traj_dcd_run}")
print(f"Log CSV:    {log_csv}")
print(f"Checkpoint: {chk_file}")
print(f"Final PDB:  {final_pdb}")

log_fh.close()


# In[13]:


# 可选：快速检查输出文件
for p in [
    out_dir / 'bpti_solvated_start.pdb',
    out_dir / 'bpti_T300_state.csv',
    out_dir / 'bpti_T300_traj.dcd',
    out_dir / 'bpti_T300.chk',
    out_dir / 'bpti_T300_final.pdb',
]:
    print(p.name, 'exists:', p.exists(), 'size:', p.stat().st_size if p.exists() else 0)


# In[14]:


import MDAnalysis as mda
from MDAnalysis.analysis import align

top = "/home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit/bpti_solvated_start.pdb"
traj = ["/home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit/bpti_T300_traj.dcd",
        "/home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit/bpti_T300_traj_resume.dcd"]

tmp_align = "aligned_tmp.dcd"

out_traj = "/home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit/bpti_T300_traj_nowat_align.dcd"
out_pdb = "/home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit/bpti_T300_nowat_align.pdb"

# 读取
u = mda.Universe(top, traj)
ref = mda.Universe(top, traj)

# 参考第一帧
ref.trajectory[0]

# 直接输出对齐后的轨迹
align.AlignTraj(
    u,
    ref,
    select="protein and name CA",
    filename=tmp_align,
    in_memory=False,
    verbose=True
).run()

# 重新读取已经对齐好的轨迹
u_align = mda.Universe(top, tmp_align)

# 去水去离子
strip_sel = "not (resname HOH WAT H2O SOL TIP3 TIP3P NA CL K MG CA ZN)"
ag = u_align.select_atoms(strip_sel)

# 输出最终轨迹
with mda.Writer(out_traj, n_atoms=ag.n_atoms) as w:
    for ts in u_align.trajectory:
        w.write(ag)

# 输出拓扑
u_align.trajectory[0]
ag.write(out_pdb)

print("Done")
