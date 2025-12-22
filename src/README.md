# sc_connectome_trajectories

## data

### sMRI
subjects without processed freesurfer: /ibmgpfs/cuizaixu_lab/Public_Data/ABCD_20221009/rawdata/sMRI

### run freesurfer
N=$(find /GPFS/cuizaixu_lab_permanent/xuhaoshu/ABCD/raw_data/smri -type f -name "*_T1w.nii" | wc -l)
sbatch --array=1-${N} src/preprocess/freesurfer.sh

## run morph
export ATLAS_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/atlas"
export OUTPUT_DIR="/ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/morphology/4YearFollowUpYArm1/SIEMENS"

sbatch --export=ALL,SUBJECTS_DIR="/GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/baselineYear1Arm1/SIEMENS/site14",ATLAS_DIR="${ATLAS_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",SITE_NAME="site14" --array=1-100 run_schaefer400_morphology.sh /path/to/sublist.txt

mkdir -p /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/sublist_by_site/4YearFollowUpYArm1
bash submit_schaefer400_morphology_all_sites.sh /GPFS/cuizaixu_lab_permanent/xuxiaoyu/ABCD/processed/freesurfer/4YearFollowUpYArm1/SIEMENS /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/sublist_by_site/4YearFollowUpYArm1 ./run_schaefer400_morphology.sh

## summarize morph
python src/preprocess/export_morphology_tables.py \
  --morph_root /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/morphology \
  --subject_info_sc /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/subject_info_sc.csv \
  --out_success /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/subject_info_morphology_success.csv \
  --out_missing /ibmgpfs/cuizaixu_lab/xuhaoshu/projects/sc_connectome_trajectories/data/ABCD/table/subject_info_sc_without_morphology.csv