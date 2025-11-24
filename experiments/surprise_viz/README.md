# Visualizing Spatio-Temporal Surprise in Video Models

**Objective**: We aim to visualize and localize surprise in video models when an impossible event occurs. 

Instructions:

(1) Generate the spatio-temporal surprise. Required: `compute_metrics=false`, `surprise=*_dense`, `tags=[*]`, other args are configurable as needed.
```sh
uv run intphyseval/main_v3.py \
    tags="[intphys2, jepa2, vith]" compute_metrics=false surprise=l1_dense context_length=16 mask_as_bool=false \
    data=intphys2 data.root=/mnt/sdb1/datasets/intphys2 data.split=Debug data.transform.crop_size=256 data.batch_size=16 data.property=permanence \
    model=jepa2/vith model.net.num_frames=32  model.ckpt_path=/mnt/sdb1/checkpoints/intphys/jepa2_vith.pt \

```

This will dump the losses.pth under the output directory printed. Example: `./logs/intuitive-physics-eval/runs/2025-11-18_00-56-50`

(2) Create visualizations.
```sh
uv run experiments/surprise_viz/main.py --run_dir ./logs/intuitive-physics-eval/runs/2025-11-18_00-56-50
```

This will create a folder under the run_dir name at `experiments/surprise_viz/outputs` with the animated `webp` visualizations.
