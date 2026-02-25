# corn_yield_xai

## Overview
`corn_yield_xai` predicts maize (corn) yield using machine learning and uncovers drivers of yield prediction using interpretable AI (XAI).  
Includes preprocessing, model training, evaluation, and XAI visualizations.

---

## Singularity and Usage

### Singularity Container
/software/setonix/2024.05/containers/sif/quay.io/pawsey/pytorch/2.2.0-rocm5.7.3/quay.io-pawsey-pytorch-2.2.0-rocm5.7.3.sif

### Usage 

### Interactive
1. Connect to a compute node and allocate resources:
salloc --account=<account> --cpus-per-task=36 --mem=25GB --nodes=1 --partition=highmem --time=5:00:00

2. Load modules and activate Python environment:
module load pytorch/2.2.0-rocm5.7.3
bash
source <path_to_python_env>/bin/activate

3. Run Jupyter Notebook:
jupyter notebook --no-browser --port=7777 --ip=0.0.0.0

4. Access in local browser via SSH tunnel:
ssh -N -L 7777:<compute_node_name>:7777 <username>@setonix.pawsey.org.au

---

### Script Submission (Scheduler)
1. Load modules and activate Python environment:
module load pytorch/2.2.0-rocm5.7.3
source <path_to_python_env>/bin/activate

2. Submit experiment script:
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK \
bash -c "source <path_to_python_env>/bin/activate && python3 scripts/run_experiment.py"

> Notes: $SLURM_JOB_NUM_NODES, $SLURM_NTASKS, and $SLURM_CPUS_PER_TASK are set by SLURM.

---

## Data
- - Genome 2 Fields Dataset: [Link](https://drive.google.com/drive/folders/1IQ3zoxBuuSP9KBYbH5dxicEjovyDtpsu)
- Preprocessing included in scripts

---

## License
MIT License