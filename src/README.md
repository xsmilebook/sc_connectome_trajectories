# sc_connectome_trajectories

## run morph
export ATLAS_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/atlas"
export OUTPUT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/morphology/4YearFollowUpYArm1/SIEMENS"

sbatch --export=ALL,SUBJECTS_DIR="/GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/baselineYear1Arm1/SIEMENS/site14",ATLAS_DIR="${ATLAS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",SITE_NAME="site14" --array=1-100 run_schaefer400_morphology.sh /path/to/sublist.txt

mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/sublist_by_site/4YearFollowUpYArm1
bash submit_schaefer400_morphology_all_sites.sh /GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/4YearFollowUpYArm1/SIEMENS /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/sublist_by_site/4YearFollowUpYArm1 ./run_schaefer400_morphology.sh
