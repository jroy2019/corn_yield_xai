# corn_yield_xai

## Overview
`corn_yield_xai` predicts maize yield using machine learning and uncovers drivers of yield prediction using Shapley Additve exPlanations (SHAP)

Repo includes Data pre-processing, model training (Random Forest and Neural Network), optiisation and evaluation, as-well XAI visualisations (local and global explanations).

---

## Singularity and Usage

### Singularity Container
[add info] idk.

### Usage 

### Interactive
1. Connect to a compute node and allocate resources:
salloc --account=<account> --cpus-per-task=x --mem=y --nodes=1 --partition=z --time=5:00:00

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
bash -c "source <path_to_python_env>/bin/activate && python3 path/to/script/script.py"

> Notes: $SLURM_JOB_NUM_NODES, $SLURM_NTASKS, and $SLURM_CPUS_PER_TASK are set by SLURM.

---

## Data
- Genome 2 Fields Dataset: [Link](https://drive.google.com/drive/folders/1IQ3zoxBuuSP9KBYbH5dxicEjovyDtpsu)

---

## License
MIT License