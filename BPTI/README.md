# BPTI pipeline

`run_bpti_pipeline_after_6h.sh` is the delayed end-to-end BPTI workflow. It waits first, then runs the 500 K generator, the 500 K to 300 K conditional flow, sampling, OpenMM relaxation/filtering, and GREX.

## Run

From any directory:

```bash
nohup /home/dell/workstations/hsj_bgflow/experiment/BPTI/run_bpti_pipeline_after_6h.sh \
  > /home/dell/workstations/hsj_bgflow/experiment/BPTI/logs/bpti_pipeline_after_6h_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The script currently uses:

```bash
sleep 5h
export CUDA_VISIBLE_DEVICES=1,2,3,4
torchrun --nproc_per_node=4 bpti-GF-bbp.py
torchrun --nproc_per_node=4 BPTI-CF-bbp.py
```

It then writes `BPTI_BG_samples.xtc`, relaxes and filters into `BPTI_BG_samples_optimized.xtc`, and runs `GREX.py`.

## Inputs

- 500 K topology: `md_T500_50ns_explicit_5PTI/bpti_T500_nowat_align.pdb`
- 500 K trajectory: `md_T500_50ns_explicit_5PTI/bpti_T500_traj_nowat_align.dcd`
- 300 K RMSD reference: `md_T300_10us_explicit_5PTI/bpti_T300_nowat_align.pdb`

## Outputs

- GF checkpoint: `models/BPTI_1UAO_AA_NLL_T500_ddp.pic`
- CF checkpoint: `models/BPTI_CF_bbp_ddp.pth`
- Raw BG samples: `BPTI_BG_samples.xtc`
- Relaxed and RMSD-filtered samples: `BPTI_BG_samples_optimized.xtc`
- GREX trajectory: `GREX_BPTI-1.xtc`
- GREX reservoir: `GREX_BPTI_reservoir_dlogp_minus_energy2.pt`

The optimization step uses `--rmsd-threshold 0.5` nm on `name CA`. Lower this value for stricter structural filtering.

## Common edits

- Change delay: edit `sleep 5h`.
- Change GPU set: edit `CUDA_VISIBLE_DEVICES` and keep `torchrun --nproc_per_node` consistent with the number of visible GPUs.
- Resume CF training: run `RESUME=1 torchrun --nproc_per_node=4 BPTI-CF-bbp.py`.
- Short smoke run: reduce `N_EPOCHS`, `--n-samples`, GREX `--target-ns`, and use temporary output/checkpoint paths.
