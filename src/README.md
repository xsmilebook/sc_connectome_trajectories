# sc_connectome_trajectories

## run morph
export SUBJECTS_DIR="/GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/baselineYear1Arm1/SIEMENS/site14"
export ATLAS_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/atlas"
export OUTPUT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/freesurfer"
* note: --array=1-100 is subjects number *
sbatch --array=1-2 run_schaefer400_morphology.sh /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/sublist/sublist_test.txt
