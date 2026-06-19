# [ECCV 2026] REFINE: Super-efficient 3D Gaussian Splatting Pruning via Rendering-Free Primitive Importance

This repository contains the official implementation of **REFINE**, a highly accelerated 3D Gaussian Splatting (3DGS) pruning framework. Our method acts as a plug-and-play, purely post-processing pipeline that can be seamlessly applied to any pre-trained 3DGS model. 

By bypassing costly forward rendering passes entirely, REFINE leverages an analytically approximated Hessian field to extract physical statistical features. This allows it to achieve an unprecedented 3,000× reduction in pruning-related computational complexity compared to state-of-the-art rendering-based methods, while maintaining highly competitive rendering quality.

---

## 1. Prerequisites & Installation

Our pruning script is lightweight and only requires a standard Python environment with PyTorch. 

**Install dependencies for pruning:**
```bash
pip install torch numpy plyfile
```

**Official 3DGS Framework:**
Since REFINE is a post-processing tool, you will need the official 3D Gaussian Splatting repository to render and evaluate the pruned models. 
* Please clone and install the official repository: [3D Gaussian Splatting GitHub](https://github.com/graphdeco-inria/gaussian-splatting).

---

## 2. Datasets & Pre-trained Models

To evaluate REFINE under a zero-shot condition (no fine-tuning after pruning), we recommend using the pre-trained models provided by the official 3DGS authors.

* **Download Pre-trained Models (14 GB):** [Official 3DGS Pre-trained Models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)
* Ensure you also have the corresponding source datasets (Mip-NeRF 360, Tanks & Temples, Deep Blending) downloaded as required by the official 3DGS pipeline.

---

## 3. Step-by-Step Usage Guide

### Step 1: Pruning the 3DGS Model
Run the `REFINE_pruning.py` script on a trained 3DGS `.ply` file. You must provide the original camera `cameras.json` to calculate view-dependent features. 

```bash
python REFINE_pruning.py \
  --start_pointcloud /path/to/pretrained/point_cloud.ply \
  --json_path /path/to/cameras.json \
  --output_path /path/to/output_folder \
  --prune_percent 0.5 
```

* **Note:** The `--prune_percent 0.5` argument means 50% of the Gaussians will be removed. 
* The script will output the memory reduction, processing throughput, and the purely scene-adaptive weights directly to your console.

### Step 2: Rendering the Pruned Model
Once the model is pruned, use the `render.py` script from the **official 3DGS repository** to render the 2D image sequences.

```bash
python render.py \
  -m /path/to/output_folder \
  -s /path/to/source_dataset
```

* `-m`: The directory containing your newly pruned `point_cloud.ply`.
* `-s`: The path to the original source dataset (COLMAP format) used to train the model.

### Step 3: Quantitative Evaluation
To calculate the PSNR, SSIM, and LPIPS metrics, use the `metrics.py` script from the **official 3DGS repository**.

```bash
python metrics.py -m /path/to/output_folder
```

This will automatically compare your rendered frames against the ground truth dataset and provide the final quality metrics.





<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{chen2026refine,
  author = {Chen, Zhang and Wan, Shuai and Yu, Mengting and Yang, Fuzheng and Hou, Junhui},
  title = {REFINE: Super-efficient 3D Gaussian Splatting Pruning via Rendering-Free Primitive Importance},
  journal = {European Conference on Computer Vision (ECCV)},
  year = {2026},
}</code></pre>
  </div>
</section>

