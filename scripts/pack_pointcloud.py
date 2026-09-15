#!/usr/bin/env python3
"""Pack point clouds into the binary format the web viewer in docs/index.html reads.

    python scripts/pack_pointcloud.py --src ~/Downloads/liseer_web_pointclouds
    python scripts/pack_pointcloud.py --demo          # from the synthetic fixture

Input: one directory per scene, one ``.npz`` per view, plus an optional
``metadata.json``.  Two npz layouts are accepted:

* the delivered layout -- ``points_q`` int16 [N,3], ``quant_scale`` float32 [3],
  ``quant_offset`` float32 [3], ``labels`` uint8 [N] (``labels`` may be absent when
  the dataset has no semantic annotation);
* a plain layout -- ``points`` float32 [N,3] and optional ``labels``.

Output::

    docs/assets/pointcloud/manifest.json
    docs/assets/pointcloud/<scene_id>/<view_id>.bin

Each ``.bin`` is interleaved ``int16`` with a stride of 4 -- x, y, z, label -- and
no header, eight bytes per point.  World coordinates come back as
``value * scale + offset`` with the per-axis ``scale``/``offset`` from the manifest.
A delivered cloud is already quantised, so it is passed through untouched.  Nothing
is thinned by default: the viewer renders a cell only when it changes, so a
600k-point dense reconstruction costs nothing while the page sits still.  Pass
``--max-dense``/``--max-other`` to cap a view if page weight matters more.

Every manifest entry also carries a ``version``, a short content hash the viewer
appends to the URL, so a re-exported cloud is never served from a stale browser
cache under an unchanged path.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

VIEW_ORDER = ['real', 'dense', 'spinning',
              'risley_0.1', 'risley_0.2', 'risley_0.5', 'risley_0.8']
VIEW_LABELS = {
    'real': 'Real scan',
    'dense': 'Dense reconstruction',
    'spinning': 'Spinning LiDAR',
    'risley_0.1': 'Prism prototype, T=0.1 s',
    'risley_0.2': 'Prism prototype, T=0.2 s',
    'risley_0.5': 'Prism prototype, T=0.5 s',
    'risley_0.8': 'Prism prototype, T=0.8 s',
}
VIEW_KIND = {'real': 'measured', 'dense': 'reconstructed',
             'spinning': 'simulated', 'risley_0.1': 'simulated',
             'risley_0.2': 'simulated', 'risley_0.5': 'simulated',
             'risley_0.8': 'simulated'}
MINUS = '−'          # a real minus sign reads better than a hyphen


def sensor_lines(row):
    """Short and long descriptions of the sensor that produced a view."""
    sp, rs = row.get('spinning'), row.get('risley')
    if sp:
        lo = (MINUS + f"{abs(sp['fov_lower_deg']):g}") if sp['fov_lower_deg'] < 0 \
            else f"+{sp['fov_lower_deg']:g}"
        hi = f"+{sp['fov_upper_deg']:g}" if sp['fov_upper_deg'] >= 0 \
            else MINUS + f"{abs(sp['fov_upper_deg']):g}"
        short = (f"{sp['beams']}×{sp['horizontal_resolution']} · "
                 f"FoV {lo}…{hi}° · h {sp['height_m']:g} m")
        full = (f"{sp['beams']} beams × {sp['horizontal_resolution']} azimuth steps, "
                f"vertical field of view {lo}° to {hi}°, "
                f"mounting height {sp['height_m']:g} m, "
                f"beam concentration {sp['beam_concentration']:g}")
        return short, full
    if rs:
        short = (f"T={rs['t_int']:g} s · {rs['num_rays']:,} rays · "
                 f"FoV {rs['fov_deg']:g}° · h {rs['height_m']:g} m")
        full = (f"integration time {rs['t_int']:g} s, {rs['num_rays']:,} rays, "
                f"circular field of view {rs['fov_deg']:g}°, "
                f"mounting height {rs['height_m']:g} m, "
                f"f1 = {rs['f1_hz']:g} Hz, "
                f"f2/f1 = {format(rs['freq_ratio'], '.6f').replace('-', MINUS)}, "
                f"rho = {rs['rho_deg']:g}°")
        return short, full
    return None, None


DATASET_LABELS = {'semantickitti': 'SemanticKITTI', 'sk': 'SemanticKITTI',
                  'nuscenes': 'nuScenes', 'nu': 'nuScenes',
                  'waymo': 'Waymo'}
# Paper order, so the rows read the same way as the appendix panels.
SCENE_ORDER = ['sk', 'nu', 'waymo']


def read_view(path):
    """Return ``(points_q int16[N,3], scale[3], offset[3], labels or None)``."""
    blob = np.load(path)
    if 'points_q' in blob:
        pts = np.asarray(blob['points_q']).astype(np.int32)
        scale = np.asarray(blob['quant_scale'], dtype=np.float64).reshape(3)
        offset = np.asarray(blob['quant_offset'], dtype=np.float64).reshape(3)
    else:
        xyz = np.asarray(blob['points'], dtype=np.float64)
        offset = xyz.min(axis=0)
        span = np.maximum(xyz.max(axis=0) - offset, 1e-6)
        scale = span / 32000.0
        pts = np.rint((xyz - offset) / scale).astype(np.int32)
    labels = (np.asarray(blob['labels']).astype(np.int32)
              if 'labels' in blob.files else None)
    return pts, scale, offset, labels


def thin(pts, labels, max_points):
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts, labels
    step = pts.shape[0] / max_points
    idx = (np.arange(max_points) * step).astype(np.int64)
    return pts[idx], (None if labels is None else labels[idx])


def content_version(path):
    """Short content hash, appended to the URL so a changed file cannot be served
    from a stale browser cache."""
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:10]


def write_bin(path, pts, labels):
    if np.abs(pts).max() > 32767:
        raise ValueError(f'{path}: quantised coordinates exceed int16')
    out = np.empty((pts.shape[0], 4), dtype='<i2')
    out[:, :3] = pts.astype('<i2')
    out[:, 3] = 0 if labels is None else np.clip(labels, 0, 255).astype('<i2')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.tofile(path)
    return out.shape[0]


def build_scenes(src, out_dir, max_dense, max_other):
    meta_rows = {}
    scene_meta = {}
    meta_path = os.path.join(src, 'metadata.json')
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        for sc in meta.get('scenes', []):
            scene_meta[sc['scene_id']] = sc
            for v in sc.get('views', []):
                meta_rows[(sc['scene_id'], v.get('view_id'))] = v

    def scene_rank(scene_id):
        prefix = scene_id.split('_')[0]
        return (SCENE_ORDER.index(prefix) if prefix in SCENE_ORDER
                else len(SCENE_ORDER), scene_id)

    scenes = []
    for scene_id in sorted((d for d in os.listdir(src)
                            if os.path.isdir(os.path.join(src, d))),
                           key=scene_rank):
        scene_dir = os.path.join(src, scene_id)
        files = {os.path.splitext(f)[0]: f for f in os.listdir(scene_dir)
                 if f.endswith(('.npz',))}
        if not files:
            continue
        order = [v for v in VIEW_ORDER if v in files] + \
                sorted(k for k in files if k not in VIEW_ORDER)
        sm = scene_meta.get(scene_id, {})
        entries = []
        for view_id in order:
            pts, scale, offset, labels = read_view(
                os.path.join(scene_dir, files[view_id]))
            raw_n = pts.shape[0]
            cap = max_dense if view_id == 'dense' else max_other
            pts, labels = thin(pts, labels, cap)
            rel = f'{scene_id}/{view_id}.bin'
            abs_path = os.path.join(out_dir, rel)
            n = write_bin(abs_path, pts, labels)
            row = meta_rows.get((scene_id, view_id), {})
            kind = VIEW_KIND.get(view_id, '')
            note = f'{raw_n:,} points'
            if n != raw_n:
                note = f'{n:,} of {raw_n:,} points'
            if kind:
                note += ' · ' + kind
            config, config_full = sensor_lines(row)
            entries.append(dict(
                id=view_id,
                label=VIEW_LABELS.get(view_id, view_id),
                file='assets/pointcloud/' + rel,
                version=content_version(abs_path),
                points=n,
                format='int16x4',
                scale=[round(float(s), 9) for s in scale],
                offset=[round(float(o), 6) for o in offset],
                has_labels=labels is not None,
                origin=row.get('origin'),
                meta=note,
                config=config,
                config_full=config_full,
            ))
            print(f'  {scene_id}/{view_id:11s} {raw_n:>8,} -> {n:>8,} points'
                  f'{"" if labels is not None else "   (no labels)"}')
        dataset = sm.get('dataset', scene_id.split('_')[0])
        scenes.append(dict(
            id=scene_id,
            dataset=DATASET_LABELS.get(dataset.lower(), dataset),
            label=sm.get('frame_tag', scene_id.replace('_', ' ')),
            ground_z=sm.get('ground_z'),
            labels_available=any(v['has_labels'] for v in entries),
            views=entries,
        ))
    return scenes


def demo_scenes(out_dir, max_dense, max_other):
    """One scene from the synthetic fixture, for exercising the viewer offline."""
    import torch
    from tests._stub_env import load_sampler_modules
    from tests.synth_frames import make_frame

    seg, _, solid = load_sampler_modules()
    sensor = dict(LiDAR_height=[1.73, 1.73], num_of_beams=[32, 32],
                  horizontal_angular_resolution=[900, 900],
                  lower_vertical_field_of_view_bound=[-25.0, -25.0],
                  upper_vertical_field_of_view_bound=[3.0, 3.0])
    points, grid_ind, grid_ind_vox, labels = make_frame(1001)

    def xyz_of(p):
        return torch.stack([p[:, 6], p[:, 7], p[:, 5]], dim=1).numpy().astype(np.float64)

    views = [('dense', xyz_of(points), labels.numpy())]
    torch.manual_seed(11)
    model = seg.PointTPV_Seg(sensor=sensor)
    model.train()
    with torch.no_grad():
        p, _, _, l, _ = model._sampling([points.clone()], [grid_ind.clone()],
                                        [grid_ind_vox.clone()], [labels.clone()])
    views.append(('spinning', xyz_of(p[0]), l[0].numpy()))
    cfg = dict(pattern='risley', risley=dict(t_int=0.8, seed=11))
    p, _, _, l, _ = solid.sample_solidstate(
        [points.clone()], [grid_ind.clone()], [grid_ind_vox.clone()],
        [labels.clone()], cfg=cfg)
    views.append(('risley_0.8', xyz_of(p[0]), l[0].numpy()))

    entries = []
    for view_id, xyz, lab in views:
        offset = xyz.min(axis=0)
        scale = np.maximum(xyz.max(axis=0) - offset, 1e-6) / 32000.0
        pts = np.rint((xyz - offset) / scale).astype(np.int32)
        cap = max_dense if view_id == 'dense' else max_other
        pts, lab = thin(pts, lab, cap)
        rel = f'synthetic_demo/{view_id}.bin'
        abs_path = os.path.join(out_dir, rel)
        n = write_bin(abs_path, pts, lab)
        entries.append(dict(id=view_id, label=VIEW_LABELS.get(view_id, view_id),
                            file='assets/pointcloud/' + rel,
                            version=content_version(abs_path), points=n,
                            format='int16x4',
                            scale=[float(s) for s in scale],
                            offset=[float(o) for o in offset],
                            has_labels=True,
                            meta=f'{n:,} points · synthetic fixture'))
        print(f'  synthetic_demo/{view_id:11s} {n:>8,} points')
    return [dict(id='synthetic_demo', dataset='Synthetic',
                 label='synthetic fixture', labels_available=True,
                 views=entries)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None)
    ap.add_argument('--out', default='docs/assets/pointcloud')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--max-dense', type=int, default=0,
                    help='cap for the dense reconstruction view (0 = keep all)')
    ap.add_argument('--max-other', type=int, default=0,
                    help='cap for every other view (0 = keep all)')
    args = ap.parse_args()
    if not args.demo and not args.src:
        ap.error('pass --src <dir> or --demo')
    os.makedirs(args.out, exist_ok=True)

    scenes = (demo_scenes(args.out, args.max_dense, args.max_other) if args.demo
              else build_scenes(args.src, args.out, args.max_dense, args.max_other))
    manifest = dict(
        format='int16x4',
        source='Response Fig. R4 (appendix B.7); exported by S3',
        classes=None,
        scenes=scenes,
    )
    manifest.pop('classes')
    path = os.path.join(args.out, 'manifest.json')
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    total_bytes = sum(
        os.path.getsize(os.path.join(args.out, v['file'].split('assets/pointcloud/')[1]))
        for s in scenes for v in s['views'])
    total_pts = sum(v['points'] for s in scenes for v in s['views'])
    print(f'wrote {path}')
    print(f'{len(scenes)} scene(s), {sum(len(s["views"]) for s in scenes)} views, '
          f'{total_pts:,} points, {total_bytes / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
