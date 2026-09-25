#!/usr/bin/env python3
"""A1 追加项 E7-E11：coarse-8 传播精度 + 逐类点数 + 混淆矩阵。

与 A1 主脚本用**完全相同**的留出实验（同 200 帧、同 seed、同 20% 留出、
同传播算法），因此这里的 fine 结果可与 A1 的 D2/D3/D4 直接对照。

E7-E9 必须在 coarse 空间**重算混淆矩阵**，不能由 fine 的逐类 IoU 平均而来——
IoU 是非线性的，且合并类别会把 car<->truck 这类 fine 级混淆变成 coarse 级的
**正确**预测。映射表直接从 seg 训练配置里 import，不另抄一份。
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from A1_label_propagation_fidelity import (VOXEL, build_voxel_labels, propagate)

# ---- COLA 8 类映射：从 seg 训练配置直接取，避免抄错 ----
_CFG = 'projects/configs/pointocc/pointtpv_semantickitti_seg_random.py'
_ns = {}
exec(compile(open(_CFG).read().split('DG = True')[0], _CFG, 'exec'), _ns)
SEMANTICKITTI_TO_CL = _ns['SEMANTICKITTI_TO_CL']
CLASS_NAMES_DG = _ns['CLASS_NAMES_DG']

SK_NAMES = {1:'outlier',10:'car',11:'bicycle',13:'bus',15:'motorcycle',16:'on-rails',
            18:'truck',20:'other-vehicle',30:'person',31:'bicyclist',32:'motorcyclist',
            40:'road',44:'parking',48:'sidewalk',49:'other-ground',50:'building',
            51:'fence',52:'other-structure',60:'lane-marking',70:'vegetation',
            71:'trunk',72:'terrain',80:'pole',81:'traffic-sign',99:'other-object',
            252:'moving-car',253:'moving-bicyclist',254:'moving-person',
            255:'moving-motorcyclist',256:'moving-on-rails',257:'moving-bus',
            258:'moving-truck',259:'moving-other-vehicle'}


def iou_from_pairs(gt, pr, classes):
    out = {}
    for c in classes:
        i = int(((pr == c) & (gt == c)).sum())
        u = int((pr == c).sum()) + int((gt == c).sum()) - i
        if u > 0:
            out[int(c)] = i / u
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/semantickitti/sequences')
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--holdout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='review_analysis/_source_data/A1c_coarse_and_confusion.json')
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    vdir = os.path.join(a.root, a.seq, 'velodyne')
    ldir = os.path.join(a.root, a.seq, 'labels')
    files = sorted(f for f in os.listdir(vdir) if f.endswith('.bin'))[:a.frames]

    GT, PR = [], []
    for fi, f in enumerate(files):
        stem = f[:-4]
        pts = np.fromfile(os.path.join(vdir, f), np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(os.path.join(ldir, stem + '.label'), np.uint32) & 0xFFFF
        if pts.shape[0] != lab.shape[0]:
            continue
        m = lab > 0
        pts, lab = pts[m], lab[m]
        if pts.shape[0] < 100:
            continue
        idx = rng.permutation(pts.shape[0])
        nho = int(round(a.holdout * pts.shape[0]))
        ho, keep = idx[:nho], idx[nho:]
        vk, vlb = build_voxel_labels(pts[keep], lab[keep])
        pred, _ = propagate(vk, vlb, pts[ho])
        GT.append(lab[ho]); PR.append(pred)
        if (fi + 1) % 25 == 0:
            print(f'  {fi+1}/{len(files)}', flush=True)

    gt = np.concatenate(GT); pr = np.concatenate(PR)

    # ---- E10 逐类点数（fine） ----
    uf, cf = np.unique(gt, return_counts=True)
    E10 = {f'{int(k)}:{SK_NAMES.get(int(k), "?")}': int(v)
           for k, v in sorted(zip(uf, cf), key=lambda x: -x[1])}
    tot = int(gt.size)

    # ---- E11 混淆矩阵行（重点四类 + 全部行的 top-5） ----
    focus = [49, 11, 30, 80]
    E11 = {}
    for c in sorted(set(uf.tolist())):
        msk = gt == c
        n = int(msk.sum())
        if n == 0:
            continue
        vals, cnts = np.unique(pr[msk], return_counts=True)
        order = np.argsort(-cnts)
        row, acc = [], 0
        for j in order:
            p, q = int(vals[j]), int(cnts[j])
            frac = q / n
            tag = 'correct' if p == c else f'{p}:{SK_NAMES.get(p, "?")}'
            row.append(f'{tag} {frac*100:.1f}%')
            acc += 1
            if acc >= 6:
                break
        key = f'{int(c)}:{SK_NAMES.get(int(c), "?")} (n={n})'
        E11[key] = ', '.join(row)
    E11_focus = {k: v for k, v in E11.items()
                 if int(k.split(':')[0]) in focus}

    # ---- E7-E9 coarse-8 空间：先映射再重算 ----
    lut = np.zeros(int(max(SEMANTICKITTI_TO_CL) ) + 1, np.int64)
    for k, v in SEMANTICKITTI_TO_CL.items():
        lut[int(k)] = int(v)
    gtc = lut[np.clip(gt, 0, len(lut) - 1)]
    prc = lut[np.clip(pr, 0, len(lut) - 1)]
    # ignore 类：coarse 0 = 'Other'（由 unlabeled/outlier 映射而来）
    valid = gtc != 0
    E7 = float((prc[valid] == gtc[valid]).mean())
    iou_c = iou_from_pairs(gtc[valid], prc[valid], range(1, 8))
    E8 = float(np.mean(list(iou_c.values()))) if iou_c else 0.0
    E9 = {f'{k}:{CLASS_NAMES_DG[k]}': round(v, 4) for k, v in sorted(iou_c.items())}

    fine_acc = float((pr == gt).mean())
    res = dict(
        task='A1-supplement (coarse-8 accuracy + confusion)',
        status='DONE', script='review_analysis/A1c_coarse_and_confusion.py',
        frames=len(GT), seq=a.seq, holdout=a.holdout, seed=a.seed,
        holdout_points=tot,
        fine_overall_accuracy=round(fine_acc, 4),
        E7_coarse_overall_accuracy=round(E7, 4),
        E8_coarse_mIoU=round(E8, 4),
        E9_coarse_per_class_IoU=E9,
        E10_class_point_counts=E10,
        E10_class_point_frac={k: round(v / tot, 6) for k, v in E10.items()},
        E11_confusion_rows_focus=E11_focus,
        E11_confusion_rows_all=E11,
        coarse_ignore='coarse class 0 (Other) excluded from E7/E8',
        mapping_source=f'{_CFG} :: SEMANTICKITTI_TO_CL',
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ('E11_confusion_rows_all', 'E10_class_point_counts',
                                   'E10_class_point_frac')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
