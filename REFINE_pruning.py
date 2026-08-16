import os
import json
import random
import time
import argparse
from typing import Dict, List

import numpy as np
import torch
from plyfile import PlyData, PlyElement


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianModel:
    def __init__(self, path: str):
        self.plydata = PlyData.read(path)
        vertex = self.plydata.elements[0]

        # Position (xyz)
        xyz = np.stack(
            (
                np.asarray(vertex["x"]),
                np.asarray(vertex["y"]),
                np.asarray(vertex["z"]),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        self.xyz = torch.from_numpy(xyz).to(device=device)

        # Opacity: original 3DGS PLY stores opacity logits.
        opac = np.asarray(vertex["opacity"], dtype=np.float32)
        self.opacity = torch.sigmoid(torch.from_numpy(opac).to(device=device))

        # Scale: original 3DGS PLY stores log-scales.
        scale_names = [f"scale_{i}" for i in range(3)]
        try:
            scales = np.stack(
                [np.asarray(vertex[n]) for n in scale_names], axis=1
            ).astype(np.float32, copy=False)
            self.scales = torch.exp(torch.from_numpy(scales).to(device=device))
        except Exception:
            self.scales = torch.ones_like(self.xyz)

        # Color (SH DC)
        dc_names = [f"f_dc_{i}" for i in range(3)]
        try:
            dc = np.stack(
                [np.asarray(vertex[n]) for n in dc_names], axis=1
            ).astype(np.float32, copy=False)
            self.sh_dc = torch.from_numpy(dc).to(device=device)
        except Exception:
            self.sh_dc = torch.zeros_like(self.xyz)

        self.rgb = torch.clamp(self.sh_dc * 0.282 + 0.5, 0.0, 1.0)

    @torch.inference_mode()
    def get_scene_adaptive_weights(self) -> Dict[str, torch.Tensor]:
        # 1. Color feature: luminance variance
        luma = (
            self.rgb[:, 0] * 0.299
            + self.rgb[:, 1] * 0.587
            + self.rgb[:, 2] * 0.114
        )
        f_color = torch.var(luma) * 100.0

        # 2. Opacity feature: edge blurriness
        f_opa = torch.mean(4.0 * self.opacity * (1.0 - self.opacity)) * 100.0

        # 3. Geometric feature: anisotropic stretch
        max_scale = self.scales.max(dim=1).values
        min_scale = self.scales.min(dim=1).values
        f_geo = torch.mean(max_scale / (min_scale + 1e-6))

        # 4. Intrinsic sensitivity factors (tau) normalization
        w_geo_raw = torch.log(f_geo + 1e-6) / 10.73
        w_color_raw = f_color / 56.06
        w_opa_raw = f_opa / 100.67

        # 5. L1 normalization
        raw = torch.stack((w_geo_raw, w_color_raw, w_opa_raw))
        weights = raw / raw.sum().clamp_min(1e-12)

        return {
            "w_geo": weights[0],
            "w_color": weights[1],
            "w_opa": weights[2],
        }


def load_cameras(file_path: str, sample_limit: int = 64) -> List[dict]:
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get("frames", data.get("cameras", []))
    if not frames:
        return []

    if sample_limit > 0 and len(frames) > sample_limit:
        rng = random.Random(42)
        frames = rng.sample(frames, sample_limit)

    cameras = []
    for fr in frames:
        c2w = np.eye(4, dtype=np.float32)

        if "position" in fr and "rotation" in fr:
            c2w[:3, :3] = np.asarray(fr["rotation"], dtype=np.float32)
            c2w[:3, 3] = np.asarray(fr["position"], dtype=np.float32)
        elif "transform_matrix" in fr:
            c2w = np.asarray(fr["transform_matrix"], dtype=np.float32)
        else:
            continue

        cameras.append(
            {
                "pos": torch.as_tensor(
                    c2w[:3, 3], dtype=torch.float32, device=device
                )
            }
        )

    return cameras


@torch.inference_mode()
def compute_h_base_fast(
    xyz: torch.Tensor,
    opacity: torch.Tensor,
    camera_positions: torch.Tensor,
    camera_chunk: int = 16,
    point_chunk: int = 1_000_000,
) -> torch.Tensor:
    """
    H_i = opacity_i / C * sum_c 1 / (||x_i - c||^2 + 0.5)

    No depth sorting and no sqrt. Cameras and Gaussians are processed
    in chunks to exploit parallel execution while bounding memory usage.
    """
    N = xyz.shape[0]
    C = camera_positions.shape[0]

    if C == 0:
        raise ValueError("No cameras were provided.")

    camera_chunk = max(1, min(camera_chunk, C))
    point_chunk = N if point_chunk <= 0 else max(1, min(point_chunk, N))

    H_base = torch.empty(N, dtype=xyz.dtype, device=xyz.device)
    cam_norm2_all = camera_positions.square().sum(dim=1)

    for p0 in range(0, N, point_chunk):
        p1 = min(p0 + point_chunk, N)
        x = xyz[p0:p1]
        opa = opacity[p0:p1]

        x_norm2 = x.square().sum(dim=1, keepdim=True)
        h_sum = torch.zeros(p1 - p0, dtype=xyz.dtype, device=xyz.device)

        for c0 in range(0, C, camera_chunk):
            c1 = min(c0 + camera_chunk, C)
            cams = camera_positions[c0:c1]
            cam_norm2 = cam_norm2_all[c0:c1]

            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x^T*c
            d2 = torch.mm(x, cams.t())
            d2.mul_(-2.0)
            d2.add_(x_norm2)
            d2.add_(cam_norm2.unsqueeze(0))
            d2.clamp_min_(1e-6)
            d2.add_(0.5)
            d2.reciprocal_()

            h_sum.add_(d2.sum(dim=1))

        H_base[p0:p1] = opa * (h_sum / float(C))

    return H_base


@torch.inference_mode()
def compute_rahd_scores_fast(
    model: GaussianModel,
    cameras: List[dict],
    dynamic_weights: Dict[str, torch.Tensor],
    camera_chunk: int = 16,
    point_chunk: int = 1_000_000,
) -> torch.Tensor:
    camera_positions = torch.stack([cam["pos"] for cam in cameras], dim=0)
    opacity = model.opacity.squeeze(-1) if model.opacity.ndim > 1 else model.opacity

    # 1. View-dependent base importance, O(N*C), no depth sorting.
    H_base = compute_h_base_fast(
        xyz=model.xyz,
        opacity=opacity,
        camera_positions=camera_positions,
        camera_chunk=camera_chunk,
        point_chunk=point_chunk,
    )

    # 2. Geometry importance: product of the two largest positive scales.
    s0 = model.scales[:, 0]
    s1 = model.scales[:, 1]
    s2 = model.scales[:, 2]
    projected_area = torch.maximum(
        s0 * s1,
        torch.maximum(s0 * s2, s1 * s2),
    )
    S_geo = H_base * projected_area

    # 3. Color importance.
    luma = (
        model.rgb[:, 0] * 0.299
        + model.rgb[:, 1] * 0.587
        + model.rgb[:, 2] * 0.114
    )
    S_color = H_base * luma.square()

    # 4. Opacity importance.
    S_opa = H_base * opacity.square()

    # 5. Attribute-level normalization.
    S_geo = S_geo / (S_geo.mean() + 1e-8)
    S_color = S_color / (S_color.mean() + 1e-8)
    S_opa = S_opa / (S_opa.mean() + 1e-8)

    # 6. Scene-adaptive weighted final score.
    return (
        dynamic_weights["w_geo"] * S_geo
        + dynamic_weights["w_color"] * S_color
        + dynamic_weights["w_opa"] * S_opa
    )


@torch.inference_mode()
def build_exact_prune_mask(
    scores: torch.Tensor,
    prune_ratio: float,
) -> torch.Tensor:
    """Build an exact-size keep mask without moving scores to CPU."""
    N = scores.numel()
    num_remove = max(0, min(int(N * prune_ratio), N))
    num_keep = N - num_remove

    if num_remove == 0:
        return torch.ones(N, dtype=torch.bool, device=scores.device)

    if num_keep == 0:
        return torch.zeros(N, dtype=torch.bool, device=scores.device)

    # Select the smaller side to reduce selection work.
    if num_remove <= num_keep:
        remove_idx = torch.topk(
            scores,
            k=num_remove,
            largest=False,
            sorted=False,
        ).indices
        mask = torch.ones(N, dtype=torch.bool, device=scores.device)
        mask[remove_idx] = False
    else:
        keep_idx = torch.topk(
            scores,
            k=num_keep,
            largest=True,
            sorted=False,
        ).indices
        mask = torch.zeros(N, dtype=torch.bool, device=scores.device)
        mask[keep_idx] = True

    return mask


def prune_and_save(
    ply_path: str,
    cam_path: str,
    output_path: str,
    prune_ratio: float,
    camera_limit: int = 64,
    camera_chunk: int = 16,
    point_chunk: int = 1_000_000,
) -> None:
    if not 0.0 <= prune_ratio <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")

    # Loading is intentionally excluded from pruning time.
    model = GaussianModel(ply_path)
    cameras = load_cameras(cam_path, sample_limit=camera_limit)
    if not cameras:
        raise RuntimeError(f"No cameras loaded from: {cam_path}")

    # Time only the REFINE pruning computation:
    # scene-adaptive weights + importance score + pruning selection.
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        dynamic_weights = model.get_scene_adaptive_weights()
        scores = compute_rahd_scores_fast(
            model=model,
            cameras=cameras,
            dynamic_weights=dynamic_weights,
            camera_chunk=camera_chunk,
            point_chunk=point_chunk,
        )
        mask = build_exact_prune_mask(scores, prune_ratio)

        end_event.record()
        end_event.synchronize()
        pruning_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = time.perf_counter()

        dynamic_weights = model.get_scene_adaptive_weights()
        scores = compute_rahd_scores_fast(
            model=model,
            cameras=cameras,
            dynamic_weights=dynamic_weights,
            camera_chunk=camera_chunk,
            point_chunk=point_chunk,
        )
        mask = build_exact_prune_mask(scores, prune_ratio)

        pruning_time = time.perf_counter() - start_time

    # Saving is intentionally excluded from pruning time.
    mask_cpu = mask.detach().cpu().numpy()
    raw_vertex_data = model.plydata.elements[0].data
    new_vertex_element = PlyElement.describe(raw_vertex_data[mask_cpu], "vertex")

    final_output_path = output_path
    if not final_output_path.lower().endswith(".ply"):
        os.makedirs(final_output_path, exist_ok=True)
        final_output_path = os.path.join(final_output_path, "point_cloud.ply")
    else:
        os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)

    PlyData([new_vertex_element], text=False).write(final_output_path)

    # Keep console output intentionally minimal.
    print(f"Processing Time: {pruning_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast REFINE 3DGS Pruning Script"
    )
    parser.add_argument("--start_pointcloud", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prune_percent", type=float, default=0.5)

    # Kept for compatibility with the original command line interface.
    parser.add_argument("--white_background", action="store_true")

    parser.add_argument("--camera_limit", type=int, default=64)
    parser.add_argument("--camera_chunk", type=int, default=16)
    parser.add_argument("--point_chunk", type=int, default=1_000_000)

    args = parser.parse_args()

    prune_and_save(
        ply_path=args.start_pointcloud,
        cam_path=args.json_path,
        output_path=args.output_path,
        prune_ratio=args.prune_percent,
        camera_limit=args.camera_limit,
        camera_chunk=args.camera_chunk,
        point_chunk=args.point_chunk,
    )
