#!/usr/bin/env bash
set -euo pipefail

cd /home/dell/workstations/hsj_bgflow/experiment/BPTI
export CUDA_VISIBLE_DEVICES=1,2,3,4
export PATH=/home/dell/software/miniforge3/envs/bgflow/bin:$PATH

torchrun --nproc_per_node=4 bpti-GF-bbp.py

torchrun --nproc_per_node=4 BPTI-CF-bbp.py

python sample_bpti_to_xtc.py \
  --n-samples 10000 \
  --batch-size 100 \
  --device cuda:1 \
  --output BPTI_BG_samples.xtc

python optimize_bpti_xtc.py \
  --input BPTI_BG_samples.xtc \
  --output BPTI_BG_samples_optimized.xtc \
  --energy-threshold -100 \
  --platform CUDA \
  --devices 0,1,2,3 \
  --rmsd-reference /home/dell/workstations/hsj_bgflow/experiment/BPTI/md_T300_10us_explicit_5PTI/bpti_T300_nowat_align.pdb \
  --rmsd-threshold 0.5 \
  --progress-interval 10

python GREX.py \
  --mode both \
  --sample BPTI_BG_samples_optimized.xtc \
  --sample-top md_T500_50ns_explicit_5PTI/bpti_T500_nowat_align.pdb \
  --output GREX_BPTI-2.xtc \
  --target-ns 100 \
  --device cuda:3 \
  --cuda-device-index 3 \
  --reservoir-file GREX_BPTI_reservoir_neg_energy2.pt \
  --mc-stride 1000 \
  --seed 20260602 \
  --logw-mode neg_energy2 \
  --minimize-initial
