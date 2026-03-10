@echo off

call conda activate mi4l
cd "C:\Users\alexn\Documents\Python Projects\mi4l_cv_pipeline"

echo Running Knee Flexion Right
python scripts/run_mi4l.py --arom "data/arom/knee_flex_ra.mp4" --prom "data/prom/knee_flex_rp.mp4" --out "results/run_kneeFlex1_right" --pose "kneeling_knee_flexion" --config "configs/default.yaml"

echo Running Trunk Extension
python scripts/run_mi4l.py --arom "data/arom/trunk_a.mp4" --prom "data/prom/trunk_p.mp4" --out "results/trunk_extension" --pose "prone_trunk_extension" --config "configs/default.yaml"

echo Running Hip Abduction
python scripts/run_mi4l.py --pose standing_hip_abduction --arom data/arom/legRaise_ra.mp4 --prom data/prom/legRaise_rp.mp4 --out results/run_standing_hip_abduction --config configs/default.yaml

echo Running Bilateral Straddle
python scripts/run_mi4l.py --pose bilateral_leg_straddle --arom data/arom/legSpread_arom.mp4 --prom data/prom/legSpread_prom.mp4 --out results/run_bilateral_leg_straddle --config configs/default.yaml

echo Running Unilateral Hip Extension
python scripts/run_mi4l.py --pose unilateral_hip_extension --arom data/arom/hip_extension_ra.mp4 --prom data/prom/hip_extension_rp.mp4 --out results/unilateral_hip_extension --config configs/default.yaml

echo Running Shoulder Stick Pass Through
python scripts/run_mi4l.py --arom data/other/stickShoulder.mp4 --pose shoulder_stick_pass_through --out results/test --config configs/default.yaml

echo Running Shoulder Flexion
python scripts/run_mi4l.py --arom "data/arom/shoulder_la.mp4" --prom "data/prom/shoulder_prom.mp4" --out "results/shoulder_flexion_001/" --pose shoulder_flexion --config configs/default.yaml

echo Done.
pause