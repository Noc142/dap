#!/usr/bin/env python3
"""
ply_to_png.py

Generate a 2D preview PNG from a PLY point cloud (headless-friendly).

Usage:
  python ply_to_png.py --in pts/5.ply --out pts/5_preview.png --max-points 200000

The script downsamples large clouds, supports RGB if present, and uses matplotlib
with Agg backend so it works on servers without an X display.
"""
import argparse
import numpy as np
from plyfile import PlyData
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_ply_vertices(path):
    ply = PlyData.read(path)
    try:
        vertex = ply['vertex'].data
    except Exception:
        raise ValueError('PLY has no vertex element')
    names = vertex.dtype.names
    if not all(n in names for n in ('x', 'y', 'z')):
        raise ValueError('vertex must contain x,y,z')
    x = vertex['x'].astype(np.float64)
    y = vertex['y'].astype(np.float64)
    z = vertex['z'].astype(np.float64)
    has_color = all(n in names for n in ('red', 'green', 'blue'))
    colors = None
    if has_color:
        colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T.astype(np.float32) / 255.0
    return x, y, z, colors


def make_preview(in_path, out_path, max_points=200000, figsize=(10, 6), dpi=150, marker_size=0.3):
    x, y, z, colors = read_ply_vertices(in_path)
    n = x.shape[0]
    if n == 0:
        raise ValueError('Empty point cloud')

    if n > max_points:
        # stratified sampling via linspace to keep structure
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    else:
        idx = np.arange(n, dtype=np.int64)

    xx = x[idx]
    yy = y[idx]
    zz = z[idx]
    cols = colors[idx] if colors is not None else 'k'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz, c=cols, s=marker_size, depthshade=False)
    ax.set_axis_off()
    # tighten the view
    try:
        ax.auto_scale_xyz([xx.min(), xx.max()], [yy.min(), yy.max()], [zz.min(), zz.max()])
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_path', help='Input PLY file',default='pts_from_npy/5.ply')
    parser.add_argument('--out', dest='out_path', help='Output PNG path', default='pts/preview/5_preview.png')
    parser.add_argument('--max-points', dest='max_points', type=int, default=200000, help='Max points to plot')
    parser.add_argument('--dpi', dest='dpi', type=int, default=150)
    parser.add_argument('--fig-w', dest='fig_w', type=float, default=10.0)
    parser.add_argument('--fig-h', dest='fig_h', type=float, default=6.0)
    args = parser.parse_args()
    # ensure output directory exists
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    make_preview(args.in_path, args.out_path, max_points=args.max_points, figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    print('Saved preview to', args.out_path)


if __name__ == '__main__':
    main()
