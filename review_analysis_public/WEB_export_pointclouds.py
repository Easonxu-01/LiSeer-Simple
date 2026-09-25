"""Export the Fig. R4 point clouds for the project website's 3-D viewer.

Three scenes (the first row of each Fig. R4 panel) x four views:
  real / dense / spinning / risley_0.8
= 12 clouds total.

Frames and sampler are the SAME code path the figure uses
(`review_analysis/A6_figR4_multi.py`), so the scenes are the figure's scenes. Two
deliberate differences from the figure, both for legibility in an orbitable
viewer rather than a thumbnail:

  * the mechanical config is FIXED and denser (see WEB_MECH) instead of the
    figure's random per-row roll;
  * the solid-state view is a single T = 0.8 s integration (the figure shows
    0.1 / 0.2 / 0.5 s).

So these clouds are NOT point-for-point identical to the printed panels; they
are the same scenes re-observed with viewer-friendly settings.

Labels are COLA-8 (0 Other / 1 Driveable Ground / 2 Structure / 3 Vehicle /
4 Nature / 5 Living Being / 6 Objects / 7 Other Ground). Note that
each dataset needs its OWN raw->COLA map: a6_figR4_multi applies the
SemanticKITTI map to every dataset, which is harmless there (its panels are
coloured by height, not by label) but would be wrong here. See LABEL_MAPS.

Coordinates: sensor frame, metres, x forward / y left / z up, origin at the
sensor. No recentring, no normalisation, no ground removal.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from review_analysis.A6_figR4_multi import (                                 # noqa: E402
    SENSOR_RANGES, _polar_feats, _read_ply_or_npy, build_full_model,
    list_frames, load_raw)
# SK coarse-class mapping is imported from the project's coarse_mapping.py (single source of truth).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'projects', 'unilidar_plugin', 'datasets'))
from coarse_mapping import SEMANTICKITTI_TO_CL as SK_TO_CL                  # noqa: E402
from review_analysis.CODE5_risley_demo import build_model, to_xyz            # noqa: E402

# --- Waymo's own taxonomy -> COLA-8, from the project's authoritative config
# (projects/configs/pointocc/pointtpv_waymo_seg_random.py: WAYMO_TO_CL).
# Waymo raw ids are 0..22 and mean entirely different things from
# SemanticKITTI's, so the SK map must NOT be reused for it.
WAYMO_TO_CL = {0: 0, 1: 3, 2: 3, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 6, 9: 6,
               10: 6, 11: 6, 12: 6, 13: 6, 14: 2, 15: 4, 16: 4, 17: 7, 18: 1,
               19: 1, 20: 1, 21: 7, 22: 7}

# --- nuScenes 32-class RAW ids -> COLA-8.
# `data/nuscenes/lidarseg_refine/` stores the 32-class ids from
# v1.0-trainval/category.json (0..31), NOT the 16-class "learning" ids. The
# project's NUSCENES_TO_CL (pointtpv_nusc_seg_random.py) is keyed on the
# 16-class ids, so applying it to these files directly would mismap. The
# table below keeps that map's semantics but is keyed on the raw ids, with
# each entry checked against the category name.
# This table is kept item-for-item identical to coarse_mapping.NUSCENES_RAW_TO_CL
# and is checked by tools/check_coarse_mapping.py (nuScenes raw-32 ids).
NUSC32_TO_CL = {
    0: 0, 29: 0, 31: 0,                           # noise, static.other, vehicle.ego -> Other
    1: 5,                                         # animal -> Living Being
    2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 5,     # human.pedestrian.* -> Living Being
    9: 6, 10: 6, 11: 6, 12: 6, 13: 6,             # movable_object.*, bicycle_rack
    14: 6, 21: 6,                                 # bicycle, motorcycle -> Objects
    15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3,     # vehicle.bus/car/construction/emergency
    22: 3, 23: 3,                                 # trailer, truck -> Vehicle
    24: 1,                                        # flat.driveable_surface
    25: 7, 26: 7, 27: 7,                          # flat.other/sidewalk/terrain -> Other Ground
    30: 4,                                        # static.vegetation -> Nature
    28: 2,                                        # static.manmade -> Structure
}
LABEL_MAPS = {'semantickitti': SK_TO_CL, 'waymo': WAYMO_TO_CL,
              'nuscenes': NUSC32_TO_CL}

# nuScenes dense labels are keyed by sample_data token, not by filename, so
# they need a lookup through v1.0-trainval/sample_data.json.
NUSC_LIDARSEG_DIR = 'data/nuscenes/lidarseg_refine/v1.0-trainval'
NUSC_LIDARSEG_RAW_DIR = 'data/nuscenes/lidarseg/v1.0-trainval'   # raw sweep
NUSC_SAMPLE_DATA = 'data/nuscenes/v1.0-trainval/sample_data.json'
_nusc_token_cache = None


def nusc_label_path(raw_path, dense=True):
    """nuScenes label file for a sweep, via its sample_data token.

    `dense=True` -> lidarseg_refine/ (one label per dense-reconstruction
    point); `dense=False` -> lidarseg/ (one per raw sweep point).
    """
    global _nusc_token_cache
    if not os.path.exists(NUSC_SAMPLE_DATA):
        return None
    if _nusc_token_cache is None:
        with open(NUSC_SAMPLE_DATA) as f:
            _nusc_token_cache = {os.path.basename(r['filename']): r['token']
                                 for r in json.load(f)}
    tok = _nusc_token_cache.get(os.path.basename(raw_path))
    if tok is None:
        return None
    d = NUSC_LIDARSEG_DIR if dense else NUSC_LIDARSEG_RAW_DIR
    p = os.path.join(d, f'{tok}_lidarseg.bin')
    return p if os.path.exists(p) else None

SCENE_IDS = {'semantickitti': 'sk_09_000771',
             'nuscenes': 'nu_1533582538648258',
             'waymo': 'waymo_0114124'}

# Mechanical config for the viewer. Fig. R4 rolls this at random per row and
# happened to draw 56 x 1461 (44k points) -- fine for a thumbnail, thin for a
# cloud you can orbit into. Fixed here at the top of each dataset's envelope
# instead, so the web view is dense enough to read. beam_concentration stays
# at the figure's 2.6: it packs the vertical angles toward the horizon, which
# is what stops the downward beams from piling into tight rings near the
# vehicle (uniform spacing puts ~78% of them inside 15 m).
WEB_MECH = dict(beams=64, hres=2048, vlow=-25.0, vup=15.0, height=1.73)
T_SOLID = 0.8


def load_labels_for(dataset, path, n):
    """COLA-8 labels, or None when the dataset has no dense labels here."""
    m = LABEL_MAPS[dataset]
    if m is None or path is None or not os.path.exists(path):
        return None
    # nuScenes lidarseg is one uint8 per point; SK/Waymo are uint32 bitfields
    # whose low 16 bits hold the semantic id.
    if dataset == 'nuscenes':
        raw = np.fromfile(path, dtype=np.uint8).astype(np.int64)
    else:
        raw = np.fromfile(path, dtype=np.uint32) & 0xFFFF
    lut = np.zeros(max(m) + 1, dtype=np.uint8)
    for k, v in m.items():
        lut[k] = v
    lab = lut[np.clip(raw, 0, len(lut) - 1)]
    if len(lab) < n:
        lab = np.pad(lab, (0, n - len(lab)))
    return lab[:n].astype(np.uint8)


def load_raw_labels(dataset, raw_path, n):
    """Per-point COLA-8 labels for the RAW sweep (not the dense cloud).

    The raw sweep has its own label file next to the .bin. SemanticKITTI
    stores one uint32 per point; Waymo stores two fields per point and the
    semantic id is the SECOND one -- verified geometrically on this frame
    (reading col 1 puts TYPE_ROAD at median z = 0.28 m and TYPE_BUILDING at
    1.73 m, i.e. buildings above the road; the [2,N] reading puts buildings
    at 0.83 m, which is impossible).
    """
    m = LABEL_MAPS[dataset]
    if m is None:
        return None
    if dataset == 'nuscenes':
        p = nusc_label_path(raw_path, dense=False)
        if p is None:
            return None
        raw = np.fromfile(p, dtype=np.uint8).astype(np.int64)
        if len(raw) != n:
            return None
        lut = np.zeros(max(m) + 1, dtype=np.uint8)
        for k, v in m.items():
            lut[k] = v
        return lut[np.clip(raw, 0, len(lut) - 1)].astype(np.uint8)
    p = (raw_path[:-4] + '.label') if dataset == 'waymo' else \
        raw_path.replace('/velodyne/', '/labels/').replace('.bin', '.label')
    if not os.path.exists(p):
        return None
    raw = np.fromfile(p, dtype=np.uint32) & 0xFFFF
    if dataset == 'waymo':
        if raw.size != 2 * n:
            return None
        raw = raw.reshape(-1, 2)[:, 1]
    if len(raw) != n:
        return None
    lut = np.zeros(max(m) + 1, dtype=np.uint8)
    for k, v in m.items():
        lut[k] = v
    return lut[np.clip(raw, 0, len(lut) - 1)].astype(np.uint8)


def voxel_downsample(xyz, labels, voxel):
    """First-point-per-voxel. Keeps labels intact (no interpolation)."""
    if voxel <= 0 or len(xyz) == 0:
        return np.arange(len(xyz))
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return np.sort(keep)


def save_view(outdir, scene_id, view, xyz, labels, voxel, quantise):
    os.makedirs(os.path.join(outdir, scene_id), exist_ok=True)
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n_full = len(xyz)
    keep = voxel_downsample(xyz, labels, voxel)
    xyz = xyz[keep]
    lab = None if labels is None else np.ascontiguousarray(labels)[keep]

    payload = {}
    if quantise:
        # int16 + scale/offset: 8 bytes/point incl. label, exact to ~1.6 mm
        # over a +-51.2 m extent. The viewer restores xyz = q*scale + offset.
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        scale = np.maximum((hi - lo) / 65534.0, 1e-9).astype(np.float32)
        q = np.round((xyz - lo) / scale).astype(np.int32) - 32767
        payload['points_q'] = q.astype(np.int16)
        payload['quant_scale'] = scale
        payload['quant_offset'] = (lo + 32767.0 * scale).astype(np.float32)
    else:
        payload['points'] = xyz
    if lab is not None:
        payload['labels'] = lab

    path = os.path.join(outdir, scene_id, f'{view}.npz')
    np.savez_compressed(path, **payload)
    return dict(path=os.path.relpath(path, outdir), num_points=int(len(xyz)),
                num_points_before_downsample=int(n_full),
                bytes=int(os.path.getsize(path)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default='delivery/liseer_web_pointclouds')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0,
                    help='must stay 0 to match the shipped Fig. R4')
    ap.add_argument('--beam-concentration', type=float, default=2.6,
                    help='the value Fig. R4 was rendered with')
    ap.add_argument('--voxel', type=float, default=0.05,
                    help='voxel size (m) for the sampled views; 0 disables')
    ap.add_argument('--voxel-dense', type=float, default=0.07,
                    help='voxel size (m) for the ~1M-point dense view, which '
                         'otherwise accounts for ~80%% of the payload')
    ap.add_argument('--no-quantise', dest='quantise', action='store_false',
                    default=True)
    ap.add_argument('--datasets', nargs='+',
                    default=['semantickitti', 'nuscenes', 'waymo'])
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(device)

    meta = dict(
        generated_from='review_analysis/WEB_export_pointclouds.py',
        figure='Response Fig. R4 (appendix B.7), first row of each panel',
        seed=args.seed, beam_concentration=args.beam_concentration,
        coordinate_frame=dict(
            frame='sensor', units='metres',
            axes='x forward, y left, z up (right-handed)',
            origin='sensor optical centre, i.e. points are already relative '
                   'to the sensor; the per-view "origin" field below is where '
                   'that sensor sits in these same coordinates',
            transforms_applied='none -- no recentring, no rotation, no '
                               'normalisation, no ground removal',
        ),
        class_names=['Other', 'Driveable Ground', 'Structure', 'Vehicle',
                     'Nature', 'Living Being', 'Objects',
                     'Other Ground'],
        quantisation=('int16 per axis; xyz = points_q.astype(f4) * '
                      'quant_scale + quant_offset' if args.quantise
                      else 'none, float32 xyz'),
        voxel_downsample_m=dict(sampled_views=args.voxel, dense_view=args.voxel_dense),
        scenes=[],
    )

    for ds in args.datasets:
        frames = list_frames(ds)
        rng = random.Random(args.seed)
        picks = rng.sample(frames, min(3, len(frames)))
        raw_p, dense_p, lab_p, tag = picks[0]          # row 0 = figure's first row
        scene_id = SCENE_IDS[ds]
        print(f'\n=== {ds}: {tag}  -> {scene_id}')

        raw_xyz = load_raw(raw_p, ds)
        dense_xyz = _read_ply_or_npy(dense_p)
        pts, gi, giv = _polar_feats(dense_xyz, device)
        # nuScenes keeps its dense labels under lidarseg_refine/, addressed by
        # sample_data token rather than by filename, so list_frames() cannot
        # supply the path and returns None for lab_p.
        if ds == 'nuscenes' and lab_p is None:
            lab_p = nusc_label_path(raw_p)
        dense_lab = load_labels_for(ds, lab_p, len(dense_xyz))
        lab_t = torch.from_numpy(
            (dense_lab.astype(np.int64) if dense_lab is not None
             else np.zeros(len(dense_xyz), dtype=np.int64))).to(device)
        base_ptr = pts.data_ptr()

        # --- mechanical: fixed dense config (see WEB_MECH) ------------------
        cfg = dict(WEB_MECH)
        full = build_full_model(device, dict(
            LiDAR_height=[cfg['height'], cfg['height']],
            num_of_beams=[cfg['beams'], cfg['beams']],
            horizontal_angular_resolution=[cfg['hres'], cfg['hres']],
            lower_vertical_field_of_view_bound=[cfg['vlow'], cfg['vlow']],
            upper_vertical_field_of_view_bound=[cfg['vup'], cfg['vup']],
        ), args.beam_concentration)
        full.return_provenance = True
        full._provenance_buffer = []
        import projects.unilidar_plugin.pointocc_model.pointtpv_seg as M
        real_u, real_c, real_b = M.random.uniform, M.random.choice, M.sample_beams
        seq = {'n': 0}

        def fake_uniform(a, b):
            seq['n'] += 1
            return {1: cfg['height'], 2: cfg['vlow'], 3: cfg['vup']}.get(
                seq['n'], real_u(a, b))

        M.random.uniform = fake_uniform
        M.random.choice = lambda s_: cfg['hres']
        M.sample_beams = lambda lo, hi: cfg['beams']
        try:
            rp, _, _, _, _ = full._sampling([pts], [gi], [giv], [lab_t])
        finally:
            M.random.uniform, M.random.choice, M.sample_beams = (
                real_u, real_c, real_b)
            full.return_provenance = False
            mech_prov = torch.cat(
                [b for _, b in full._provenance_buffer]).cpu().numpy()
            full._provenance_buffer = []
        mech_xyz = to_xyz(rp[0])
        mech_lab = None if dense_lab is None else dense_lab[mech_prov]

        ground_z = float(np.percentile(dense_xyz[:, 2], 2))
        mech_origin = [0.0, 0.0, ground_z + float(cfg['height'])]

        raw_lab = load_raw_labels(ds, raw_p, len(raw_xyz))
        views = [
            ('real', raw_xyz, raw_lab, None),
            ('dense', dense_xyz, dense_lab, None),
            ('spinning', mech_xyz, mech_lab,
             dict(origin=mech_origin, spinning=dict(
                 beams=int(cfg['beams']),
                 horizontal_resolution=int(cfg['hres']),
                 fov_lower_deg=float(cfg['vlow']),
                 fov_upper_deg=float(cfg['vup']),
                 height_m=float(cfg['height']),
                 beam_concentration=args.beam_concentration))),
        ]

        T = T_SOLID
        sp, _, _, _, inf = model.sample_solid_state(
            [pts], [gi], [giv], [lab_t], integration_time=T,
            footprint_scale=1.0, occlusion_mode='knn', seed=args.seed + 0)
        assert pts.data_ptr() == base_ptr, 'input tensor was replaced'
        info = inf[0]
        sxyz = to_xyz(sp[0])
        # The solid-state path does not carry provenance, so labels come from
        # a nearest-neighbour lookup into the dense cloud. This is exact, not
        # an approximation: every returned point IS one of the dense points,
        # only its range has been re-projected onto the ray.
        slab = None
        if dense_lab is not None:
            from scipy.spatial import cKDTree
            _, nn = cKDTree(dense_xyz).query(sxyz, k=1)
            slab = dense_lab[nn]
        views.append((
            f'risley_{T}', sxyz, slab,
            dict(origin=[0.0, 0.0, ground_z + 1.73], risley=dict(
                t_int=T, num_rays=int(info['num_rays']),
                fov_deg=70.4, height_m=1.73,
                f1_hz=7294.0, freq_ratio=-1 / 1.6180339887,
                rho_deg=17.5,
                psi1=float(info['psi1']), psi2=float(info['psi2'])))))

        scene_meta = dict(scene_id=scene_id, dataset=ds, frame_tag=tag,
                          ground_z=ground_z,
                          labels_available=dense_lab is not None,
                          views=[])
        for view, xyz, lab, extra in views:
            vox = args.voxel_dense if view == 'dense' else args.voxel
            rec = save_view(args.outdir, scene_id, view, xyz, lab,
                            vox, args.quantise)
            rec['voxel_m'] = vox
            rec['view_id'] = view
            rec['origin'] = ([0.0, 0.0, ground_z + 1.73] if extra is None
                             else extra.get('origin', [0.0, 0.0, ground_z]))
            rec['ground_z'] = ground_z
            rec['has_labels'] = lab is not None
            if extra:
                for k in ('spinning', 'risley'):
                    if k in extra:
                        rec[k] = extra[k]
            if view in ('real', 'dense'):
                rec['origin'] = [0.0, 0.0, 0.0] if view == 'real' else rec['origin']
                rec['note'] = ('raw sensor sweep as stored by the dataset'
                               if view == 'real'
                               else 'dense reconstruction, the input every '
                                    'simulated view re-observes')
            scene_meta['views'].append(rec)
            print(f'  {view:<12} {rec["num_points_before_downsample"]:>9,} '
                  f'-> {rec["num_points"]:>8,} pts  '
                  f'{rec["bytes"] / 1e6:>6.2f} MB'
                  f'{"" if rec["has_labels"] else "   (no labels)"}')
        meta['scenes'].append(scene_meta)

    with open(os.path.join(args.outdir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    total = sum(v['bytes'] for s in meta['scenes'] for v in s['views'])
    print(f'\n[web] {len(meta["scenes"])} scenes, total {total / 1e6:.2f} MB '
          f'-> {args.outdir}')


if __name__ == '__main__':
    main()
