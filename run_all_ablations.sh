#!/bin/bash
#SBATCH --account=msoleyma_1026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1         # Request exactly 1 L40s GPU per task
#SBATCH --time=02:00:00           # Max time per task
#SBATCH --cpus-per-task=8         # Matches num_workers=8 in DataLoader
#SBATCH --mem=32G                 # Prevent OOM when loading embeddings
#SBATCH --array=0-20%4             # 21 parallel jobs total

# Map each array ID to a specific script and experiment
TASKS=(
    # Section 2 (Indices 0-8)
    "sec2_dyadic.py Full_Dyadic"
    "sec2_dyadic.py No_Partner"
    "sec2_dyadic.py SelfVideo_PartnerVideo"
    "sec2_dyadic.py No_SelfVideo"
    "sec2_dyadic.py No_Audio"
    "sec2_dyadic.py No_Text"
    "sec2_dyadic.py Only_PartnerVideo"
    "sec2_dyadic.py Only_SelfVideo"
    "sec2_dyadic.py Audio_SelfVideo"
    
    # Section 3 (Indices 9-18)
    "sec3_visual.py Full_Visual_Dyadic"
    "sec3_visual.py FAU_Only"
    "sec3_visual.py Head_Only"
    "sec3_visual.py Gaze_Only"
    "sec3_visual.py Body_Only"
    "sec3_visual.py FAU_Head"
    "sec3_visual.py FAU_Gaze"
    "sec3_visual.py FAU_Body"
    "sec3_visual.py Face_Only"
    "sec3_visual.py Body_Speaker_FAU_Partner"
    
    # Section 4 (Indices 19-20)
    "sec4_fusion.py Audio_DyadicFAU"
    "sec4_fusion.py Audio_FAU_Only"
)

# Extract the script and experiment name for the current array task
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
SCRIPT=$(echo $TASK | awk '{print $1}')
EXPERIMENT=$(echo $TASK | awk '{print $2}')

echo "========================================================="
echo "Starting Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Executing: python $SCRIPT --experiment $EXPERIMENT"
echo "========================================================="

# Load the environment
module load conda
# Replace 'your_env_name' with your actual conda environment
source activate calm_conflict

# Run the task
python $SCRIPT --experiment "$EXPERIMENT"