import os
import json
import random
import time
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianModel:
    def __init__(self, path):
        print(f"Loading model from: {path}")
        self.plydata = PlyData.read(path)

        # Position (xyz)
        xyz = np.stack((np.asarray(self.plydata.elements[0]["x"]),
                        np.asarray(self.plydata.elements[0]["y"]),
                        np.asarray(self.plydata.elements[0]["z"])), axis=1)
        self.xyz = torch.tensor(xyz, dtype=torch.float32, device=device)

        # Opacity
        opac = np.asarray(self.plydata.elements[0]["opacity"])
        self.opacity = torch.sigmoid(torch.tensor(opac, dtype=torch.float32, device=device))

        # Scale
        scale_names = [f'scale_{i}' for i in range(3)]
        try:
            scales = np.stack([np.asarray(self.plydata.elements[0][n]) for n in scale_names], axis=1)
            self.scales = torch.exp(torch.tensor(scales, dtype=torch.float32, device=device))
        except Exception:
            self.scales = torch.ones_like(self.xyz)

        # Color (DC)
        dc_names = [f'f_dc_{i}' for i in range(3)]
        try:
            dc = np.stack([np.asarray(self.plydata.elements[0][n]) for n in dc_names], axis=1)
            self.sh_dc = torch.tensor(dc, dtype=torch.float32, device=device)
        except Exception:
            self.sh_dc = torch.zeros_like(self.xyz)

        self.rgb = torch.clamp(self.sh_dc * 0.282 + 0.5, 0.0, 1.0)

    def get_scene_adaptive_weights(self):
        """
        Pure scene-adaptive feature extraction. Achieves data-driven weight allocation
        by extracting physical statistical features and dividing by intrinsic sensitivity factors (τ).
        """
        N = self.xyz.shape[0]
        feature_flops = 0.0

        # 1. Color feature: Luminance variance
        luma = self.rgb[:, 0] * 0.299 + self.rgb[:, 1] * 0.587 + self.rgb[:, 2] * 0.114
        f_color = torch.var(luma).item() * 100
        feature_flops += 9 * N

        # 2. Opacity feature: Edge blurriness
        f_opa = torch.mean(4 * self.opacity * (1 - self.opacity)).item() * 100
        feature_flops += 4 * N

        # 3. Geometric feature: Anisotropic stretch
        max_scale = self.scales.max(dim=1)[0]
        min_scale = self.scales.min(dim=1)[0]
        f_geo = torch.mean(max_scale / (min_scale + 1e-6)).item()
        feature_flops += 3 * N

        # 4. Intrinsic Sensitivity Factors (τ) normalization
        w_geo_raw = np.log(f_geo + 1e-6) / 10.73
        w_color_raw = f_color / 56.06
        w_opa_raw = f_opa / 100.67

        # 5. L1 normalization
        total_w = w_geo_raw + w_color_raw + w_opa_raw
        final_w = {
            'w_geo': w_geo_raw / total_w,
            'w_color': w_color_raw / total_w,
            'w_opa': w_opa_raw / total_w
        }

        print("=" * 60)
        print(f"💡 [Pure Scene-Adaptive Intelligence Active]")
        print(f"   Scene Stats   -> Geo_Stretch: {f_geo:.1f}, Color_Var: {f_color:.1f}, Opa_Fuzzy: {f_opa:.1f}")
        print(f"   Final Weights -> Geo: {final_w['w_geo']:.3f}, Color: {final_w['w_color']:.3f}, Opa: {final_w['w_opa']:.3f}")
        print("=" * 60)

        return final_w, feature_flops

def load_cameras(file_path, sample_limit=64):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = data if isinstance(data, list) else data.get('frames', data.get('cameras', []))
    if not frames: return []

    if len(frames) > sample_limit:
        rng = random.Random(42)
        frames = rng.sample(frames, sample_limit)

    cameras = []
    flip_mat = np.eye(4)
    for fr in frames:
        c2w = np.eye(4, dtype=np.float32)
        if 'position' in fr and 'rotation' in fr:
            c2w[:3, :3] = np.array(fr['rotation'], dtype=np.float32)
            c2w[:3, 3] = np.array(fr['position'], dtype=np.float32)
        elif 'transform_matrix' in fr:
            c2w = np.array(fr['transform_matrix'], dtype=np.float32)
        else:
            continue
        c2w = c2w @ flip_mat
        cameras.append({'pos': torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)})
    return cameras

def compute_rahd_scores(model, cameras, dynamic_weights, feature_flops):
    N = model.xyz.shape[0]
    total_flops = feature_flops

    H_base = torch.zeros(N, device=device)
    calc_cams = cameras if len(cameras) > 0 else []

    for cam in calc_cams:
        vec = model.xyz - cam['pos']
        depth_sq = torch.sum(vec ** 2, dim=1)
        depth_sq = torch.clamp(depth_sq, min=1e-6)
        depth = torch.sqrt(depth_sq)

        s_depth, idx = torch.sort(depth)
        s_opac = model.opacity[idx].squeeze()

        w_depth = 1.0 / (s_depth ** 2 + 0.5)
        w_unified = s_opac * w_depth

        curr_base = torch.empty(N, device=device)
        curr_base[idx] = w_unified
        H_base += curr_base

        total_flops += 14 * N

    if len(calc_cams) > 0:
        H_base /= len(calc_cams)
        total_flops += 1 * N

    # Intrinsic Coupling
    sorted_scales, _ = torch.sort(model.scales, dim=1)
    projected_area = sorted_scales[:, -1] * sorted_scales[:, -2]
    S_geo = H_base * projected_area
    total_flops += 2 * N

    luma = model.rgb[:, 0] * 0.299 + model.rgb[:, 1] * 0.587 + model.rgb[:, 2] * 0.114
    color_energy = luma ** 2
    S_color = H_base * color_energy
    total_flops += 7 * N

    S_opa = H_base * (model.opacity.squeeze() ** 2)
    total_flops += 2 * N

    # Attribute-level normalization
    S_geo /= (S_geo.mean() + 1e-8)
    S_color /= (S_color.mean() + 1e-8)
    S_opa /= (S_opa.mean() + 1e-8)
    total_flops += 6 * N

    # Apply pure scene-adaptive weights
    final_score = (dynamic_weights['w_geo'] * S_geo) + \
                  (dynamic_weights['w_color'] * S_color) + \
                  (dynamic_weights['w_opa'] * S_opa)
    total_flops += 5 * N

    return final_score.cpu().numpy(), total_flops

def prune_and_save(ply_path, cam_path, output_path, prune_ratio):
    start_time = time.time()

    model = GaussianModel(ply_path)
    cameras = load_cameras(cam_path)

    if not cameras:
        print("Error: No cameras loaded.")
        return

    dynamic_weights, feature_flops = model.get_scene_adaptive_weights()
    scores, flops = compute_rahd_scores(model, cameras, dynamic_weights, feature_flops)

    total_points = len(scores)
    k = int(total_points * prune_ratio)

    if k >= total_points:
        threshold = float('inf')
    elif k <= 0:
        threshold = -float('inf')
    else:
        threshold = np.partition(scores, k)[k]

    mask = scores > threshold

    keep_count, remove_count = np.sum(mask), total_points - np.sum(mask)
    gflops = flops / 1e9

    print("-" * 60)
    print(f"Pure Scene-Adaptive REFINE Pruning Statistics")
    print(f"Prune Ratio:       {prune_ratio}")
    print(f"Removing:          {remove_count}")
    print(f"Keeping:           {keep_count}")
    print(f"Total Computation: {gflops:.4f} GFLOPs")
    print("-" * 60)

    raw_vertex_data = model.plydata.elements[0].data
    new_vertex_element = PlyElement.describe(raw_vertex_data[mask], 'vertex')

    final_output_path = output_path
    if not final_output_path.lower().endswith('.ply'):
        os.makedirs(final_output_path, exist_ok=True)
        final_output_path = os.path.join(final_output_path, "point_cloud.ply")
    else:
        os.makedirs(os.path.dirname(final_output_path) or '.', exist_ok=True)

    PlyData([new_vertex_element], text=False).write(final_output_path)

    elapsed_time = time.time() - start_time
    print(f"Saved pruned model to: {final_output_path}")
    print(f"Processing Time:   {elapsed_time:.4f} seconds")
    print(f"Throughput:        {gflops / elapsed_time:.2f} GFLOPs/s")

    if os.path.exists(ply_path) and os.path.exists(final_output_path):
        old_size = os.path.getsize(ply_path) / (1024 ** 2)
        new_size = os.path.getsize(final_output_path) / (1024 ** 2)
        print(f"Size reduced:      {old_size:.2f} MB -> {new_size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Scene-Adaptive REFINE 3DGS Pruning Script")
    parser.add_argument("--start_pointcloud", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prune_percent", type=float, default=0.5)
    parser.add_argument("--white_background", action="store_true")

    args = parser.parse_args()

    prune_and_save(args.start_pointcloud, args.json_path, args.output_path, args.prune_percent)