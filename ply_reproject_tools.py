"""
PLY reprojection & stereo helpers.
- Default behavior: run with hardcoded defaults below (no args needed).
- CLI overrides: provide --mode / --in-* / --out-* / --h / --w / --baseline / --write-depth.
Modes:
  reproj      : single PLY -> ERP RGB (and optional depth)
  reproj_dir  : folder PLYs -> ERP RGB folder (and optional depth folder)
  stereo      : single PLY -> left/right shifted PLYs
  stereo_dir  : folder PLYs -> left/right folders
"""
import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
import cv2


# --- Geometry helpers ---
def spherical_from_dirs(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2.0 * np.pi, theta)
    r = np.linalg.norm(dirs, axis=-1)
    phi = np.arccos(np.clip(z / np.maximum(r, 1e-8), -1.0, 1.0))
    return theta, phi


def uv_from_dirs(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta, phi = spherical_from_dirs(dirs)
    u = 1.0 - theta / (2.0 * np.pi)  # match depth2point convention
    v = phi / np.pi
    return u, v


# --- IO helpers ---
def read_ply(path: Path):
    ply = PlyData.read(path)
    vertex = ply['vertex'].data
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T.astype(np.float32)
    has_color = all(k in vertex.dtype.names for k in ('red', 'green', 'blue'))
    colors = None
    if has_color:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.uint8)
    return pts, colors


def write_ply(path: Path, pts: np.ndarray, colors: np.ndarray | None):
    pts = pts.reshape(-1, 3)
    if colors is None:
        colors = np.zeros_like(pts, dtype=np.uint8)
    colors = colors.reshape(-1, 3)
    vertex_data = np.empty(len(pts), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex_data['x'] = pts[:, 0]
    vertex_data['y'] = pts[:, 1]
    vertex_data['z'] = pts[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    PlyData([PlyElement.describe(vertex_data, 'vertex', comments=['point cloud'])]).write(path)


# --- Reprojection ---
def reproject_to_erp(ply_path: Path, out_png: Path, out_depth_png: Path | None = None, h: int = 1440, w: int = 2912):
    pts, colors = read_ply(ply_path)
    if pts.size == 0:
        raise ValueError('empty point cloud')

    r = np.linalg.norm(pts, axis=1)
    dirs = pts / np.maximum(r[:, None], 1e-8)
    u, v = uv_from_dirs(dirs)
    px = np.clip((u * w).astype(np.int32), 0, w - 1)
    py = np.clip((v * h).astype(np.int32), 0, h - 1)
    idx_flat = py * w + px

    min_depth = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(min_depth, idx_flat, r)

    color_img = np.zeros((h * w, 3), dtype=np.uint8)
    if colors is not None:
        tol = 1e-4
        mask = np.abs(r - min_depth[idx_flat]) <= tol
        chosen_idx = idx_flat[mask]
        chosen_cols = colors[mask]
        color_img[chosen_idx] = chosen_cols
    color_img = color_img.reshape(h, w, 3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

    if out_depth_png is not None:
        depth_img = min_depth.reshape(h, w)
        finite = np.isfinite(depth_img)
        if finite.any():
            d = depth_img.copy()
            d[~finite] = 0
            d_min, d_max = d[finite].min(), d[finite].max()
            if d_max > d_min:
                d_vis = ((d - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
            else:
                d_vis = np.zeros_like(d, dtype=np.uint8)
        else:
            d_vis = np.zeros_like(depth_img, dtype=np.uint8)
        cv2.imwrite(str(out_depth_png), d_vis)


def reproject_dir(in_dir: Path, out_rgb_dir: Path, out_depth_dir: Path | None, h: int = 1440, w: int = 2912):
    out_rgb_dir.mkdir(parents=True, exist_ok=True)
    if out_depth_dir is not None:
        out_depth_dir.mkdir(parents=True, exist_ok=True)
    ply_files = sorted(in_dir.glob('*.ply'))
    if not ply_files:
        print(f"no ply files in {in_dir}")
        return
    for ply_path in ply_files:
        stem = ply_path.stem
        out_png = out_rgb_dir / f"{stem}.png"
        out_depth_png = (out_depth_dir / f"{stem}_depth.png") if out_depth_dir is not None else None
        try:
            reproject_to_erp(ply_path, out_png, out_depth_png, h=h, w=w)
            print(f"reproj ok: {ply_path.name} -> {out_png.name}")
        except Exception as e:
            print(f"reproj fail: {ply_path.name}: {e}")


# --- Stereo shift ---
def translate_point_cloud(ply_path: Path, out_left: Path, out_right: Path, baseline_m: float = 0.032):
    pts, colors = read_ply(ply_path)
    shift = np.array([baseline_m, 0.0, 0.0], dtype=np.float32)
    pts_right = pts - shift
    pts_left = pts + shift
    out_left.parent.mkdir(parents=True, exist_ok=True)
    out_right.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out_left, pts_left, colors)
    write_ply(out_right, pts_right, colors)


def translate_dir(in_dir: Path, out_left_dir: Path, out_right_dir: Path, baseline_m: float = 0.032):
    out_left_dir.mkdir(parents=True, exist_ok=True)
    out_right_dir.mkdir(parents=True, exist_ok=True)
    ply_files = sorted(in_dir.glob('*.ply'))
    if not ply_files:
        print(f"no ply files in {in_dir}")
        return
    for ply_path in ply_files:
        stem = ply_path.stem
        out_left = out_left_dir / f"{stem}_left.ply"
        out_right = out_right_dir / f"{stem}_right.ply"
        try:
            translate_point_cloud(ply_path, out_left, out_right, baseline_m=baseline_m)
            print(f"stereo ok: {ply_path.name} -> {out_left.name}, {out_right.name}")
        except Exception as e:
            print(f"stereo fail: {ply_path.name}: {e}")


# --- Hardcoded defaults (edit here) ---
DEFAULT_MODE = "reproj_dir"  # reproj | reproj_dir | stereo | stereo_dir
DEFAULT_H = 512
DEFAULT_W = 1024

# single-file defaults
if DEFAULT_MODE == "reproj":
    DEFAULT_IN_FILE = Path("pts_stereo/left/5.ply")
else:  # DEFAULT_MODE == "stereo":
    DEFAULT_IN_FILE = Path("pts/5.ply")
# reproj
DEFAULT_OUT_FILE = Path("erp/rgb/5_rgb.png")
DEFAULT_OUT_DEPTH = Path("erp/depth/5_depth.png")
# warp
DEFAULT_OUT_LEFT = Path("pts_stereo/left/5_left.ply")
DEFAULT_OUT_RIGHT = Path("pts_stereo/right/5_right.ply")

# folder defaults
if DEFAULT_MODE == "reproj_dir":
    DEFAULT_IN_DIR = Path("pts_stereo/left")
else:  # DEFAULT_MODE == "stereo_dir":
    DEFAULT_IN_DIR = Path("pts")
# reproj
DEFAULT_OUT_DIR_RGB = Path("erp/rgb")
DEFAULT_OUT_DIR_DEPTH = Path("erp/depth")
# warp
DEFAULT_OUT_DIR_LEFT = Path("pts_stereo/left")
DEFAULT_OUT_DIR_RIGHT = Path("pts_stereo/right")
DEFAULT_WRITE_DEPTH = True

DEFAULT_BASELINE = 0.032


def parse_args():
    parser = argparse.ArgumentParser(description="PLY reprojection & stereo (defaults are hardcoded; args override)")
    parser.add_argument("--mode", choices=["reproj", "reproj_dir", "stereo", "stereo_dir"], default=DEFAULT_MODE)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    parser.add_argument("--baseline", type=float, default=DEFAULT_BASELINE)
    parser.add_argument("--write-depth", action="store_true", default=DEFAULT_WRITE_DEPTH)

    parser.add_argument("--in-file", type=Path, default=DEFAULT_IN_FILE)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--out-depth", type=Path, default=DEFAULT_OUT_DEPTH)
    parser.add_argument("--out-left", type=Path, default=DEFAULT_OUT_LEFT)
    parser.add_argument("--out-right", type=Path, default=DEFAULT_OUT_RIGHT)

    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    parser.add_argument("--out-rgb-dir", type=Path, default=DEFAULT_OUT_DIR_RGB)
    parser.add_argument("--out-depth-dir", type=Path, default=DEFAULT_OUT_DIR_DEPTH)
    parser.add_argument("--out-left-dir", type=Path, default=DEFAULT_OUT_DIR_LEFT)
    parser.add_argument("--out-right-dir", type=Path, default=DEFAULT_OUT_DIR_RIGHT)

    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    if mode == "reproj":
        reproject_to_erp(args.in_file, args.out_file, args.out_depth, h=args.h, w=args.w)
    elif mode == "reproj_dir":
        depth_dir = args.out_depth_dir if args.write_depth else None
        reproject_dir(args.in_dir, args.out_rgb_dir, depth_dir, h=args.h, w=args.w)
    elif mode == "stereo":
        translate_point_cloud(args.in_file, args.out_left, args.out_right, baseline_m=args.baseline)
    elif mode == "stereo_dir":
        translate_dir(args.in_dir, args.out_left_dir, args.out_right_dir, baseline_m=args.baseline)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    main()
