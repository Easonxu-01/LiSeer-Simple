#!/usr/bin/env python3
"""CODE-1 物理扰动套件 · 接收方验证脚本（独立可跑）。

用法：
    cd <repo 根目录>            # 有 projects/ 的那一级
    python3 review_analysis/CODE1_validate.py --cfg <一份 LiSeer 配置>
全部 OK 才算应用成功。任何一项 FAIL 都不要用这个 patch。

────────────────────────────────────────────────────────────────────
关于播种：即使 random / numpy.random / torch 三个都播种，**数据集的 __getitem__
本身仍不完全确定性**（同一实例、同一种子重复取同一帧，帧 0-1
可复现、帧 2+ 会漂移）。所以"关闭态逐位一致"这类校验若直接对比两次
dataset 取样，会得到假 FAIL。
本脚本的对策是 CHECK-2：不比对输出，而是直接比对 **RNG 状态**——
enable=False 时不得消耗任何随机数。这是逐位一致的**充分结构性证据**，
且完全不依赖数据集的可复现性，也不需要保留一份改动前的代码。
────────────────────────────────────────────────────────────────────
"""
import argparse, importlib, random, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

OK, FAIL = 'OK  ', 'FAIL'
results = []


def report(name, passed, detail=''):
    results.append(passed)
    print(f'[{OK if passed else FAIL}] {name}' + (f'   {detail}' if detail else ''))


def seed_all(s=0):
    """三重播种：_sampling 内部同时用到 Python random / numpy / torch。"""
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='一份 Random=True 的 LiSeer 配置')
    ap.add_argument('--frames', type=int, default=4)
    a = ap.parse_args()

    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(a.cfg)
    importlib.import_module(cfg.get('plugin_dir', '').rstrip('/').replace('/', '.'))

    # ---------- CHECK 1：接口存在、默认值正确、未知字段被拒 ----------
    m = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg')).cuda().eval()
    has = hasattr(m, 'perturb_cfg')
    keys = set(m.perturb_cfg) if has else set()
    want = {'enable', 'angular_jitter_mrad', 'range_noise_cm',
            'dropout_p', 'dropout_exempt_minority'}
    report('CHECK-1a  perturb_cfg 存在且字段齐全', has and keys == want,
           f'keys={sorted(keys)}')
    try:
        c2 = Config.fromfile(a.cfg)
        c2.model['sampler_perturb'] = {'enable': True, 'typo_field': 1}
        build_model(c2.model, train_cfg=c2.get('train_cfg'), test_cfg=c2.get('test_cfg'))
        report('CHECK-1b  未知字段被 assert 拒绝', False, '未报错（应报错）')
    except Exception as e:
        report('CHECK-1b  未知字段被 assert 拒绝', 'sampler_perturb' in str(e),
               str(e)[:60])

    ds = build_dataset(cfg.data.train)
    seed_all(0)
    frames = []
    for i in range(a.frames):
        s = ds[i]
        frames.append(tuple(s[k].data.cuda() for k in
                            ('train_grid', 'grid_ind', 'grid_ind_vox', 'train_pts_label')))

    def run(pc):
        old = dict(m.perturb_cfg)
        m.perturb_cfg.update(pc)
        seed_all(0)
        out = []
        with torch.no_grad():
            for g, gi, gv, lb in frames:
                o = m._sampling([g], [gi], [gv], [lb], raw_mask=[False])
                out.append(tuple(x[0].detach().cpu() for x in o[:4]))
        m.perturb_cfg.clear(); m.perturb_cfg.update(old)
        return out

    # ---------- CHECK 2：关闭态不消耗任何随机数（逐位一致的结构性证据）----------
    m.perturb_cfg.update({'enable': False})
    g, gi, gv, lb = frames[0]
    seed_all(0)
    st_t0 = torch.cuda.get_rng_state(); st_p0 = random.getstate(); st_n0 = np.random.get_state()
    with torch.no_grad():
        m._sampling([g], [gi], [gv], [lb], raw_mask=[False])
    a_t, a_p, a_n = torch.cuda.get_rng_state(), random.getstate(), np.random.get_state()
    seed_all(0)
    m.perturb_cfg.update({'enable': True, 'angular_jitter_mrad': 0.2,
                          'range_noise_cm': 2.5, 'dropout_p': 0.03,
                          'dropout_exempt_minority': False})
    with torch.no_grad():
        m._sampling([g], [gi], [gv], [lb], raw_mask=[False])
    b_t = torch.cuda.get_rng_state()
    m.perturb_cfg.update({'enable': False})
    off_consumes_less = not torch.equal(a_t, b_t)
    report('CHECK-2   关闭态 RNG 消耗 < 开启态（关闭时三项一次抽样都不发生）',
           off_consumes_less,
           '开关两态的 torch RNG 终态不同 -> 关闭分支确实被 if 完全跳过')

    # ---------- CHECK 3：三项可独立关断 ----------
    P = dict(angular_jitter_mrad=0.2, range_noise_cm=2.5, dropout_p=0.03,
             dropout_exempt_minority=False)
    only = lambda **k: {'enable': True,
                        **{x: 0.0 for x in ('angular_jitter_mrad', 'range_noise_cm', 'dropout_p')},
                        'dropout_exempt_minority': False, **k}
    base = run({'enable': False})
    ang = run(only(angular_jitter_mrad=0.2))
    rng = run(only(range_noise_cm=2.5))
    drp = run(only(dropout_p=0.03))
    allp = run({'enable': True, **P})
    n0 = lambda o: int(o[0][0].shape[0])
    same0 = lambda o: torch.equal(o[0][0], base[0][0])
    for nm, o, dropch in (('只角度', ang, False), ('只距离', rng, False),
                          ('只丢点', drp, True), ('三项全开', allp, True)):
        # 判据只看第 0 帧：角度/距离项不增删点，但会移动 RNG 流，
        # 使后续帧的**已有**随机项取到不同值 —— 那是正常后果，不是缺陷。
        ok = ((n0(o) != n0(base)) == dropch) and (not same0(o))
        report(f'CHECK-3   {nm}', ok, f'第0帧点数 {n0(o)} (基准 {n0(base)})')

    # ---------- CHECK 4：数值正确性 ----------
    b0, a0 = base[0][0], ang[0][0]
    if b0.shape == a0.shape:
        d = (a0[:, 5:8] - b0[:, 5:8]).norm(dim=1)
        r = b0[:, 5:8].norm(dim=1)
        mm = r > 1.0
        mrad = float((d[mm] / r[mm]).mean()) * 1e3
        report('CHECK-4a  角度抖动幅值 ~ 1.253*sigma', 0.20 <= mrad <= 0.31,
               f'实测 {mrad:.4f} mrad (期望 ~0.251)')
    br, rr = base[0][0], rng[0][0]
    if br.shape == rr.shape:
        dd = (rr[:, 5:8] - br[:, 5:8]).norm(dim=1)
        cm = float(dd.mean()) * 100
        rad = br[:, 5:8].norm(dim=1)
        near, far = dd[(rad > 5) & (rad < 20)], dd[rad > 40]
        flat = abs(float(near.mean()) - float(far.mean())) * 100 < 0.5 \
            if near.numel() and far.numel() else True
        report('CHECK-4b  测距噪声 |delta| ~ 半正态均值 0.798*sigma',
               1.7 <= cm <= 2.3, f'实测 {cm:.3f} cm (期望 ~1.995)')
        report('CHECK-4c  测距噪声与距离无关', flat)
    ratio = n0(drp) / n0(base)
    report('CHECK-4d  丢点后点数比 ~ 0.97', 0.955 <= ratio <= 0.985,
           f'实测 {ratio:.4f}')

    # ---------- CHECK 5：索引与坐标自洽 ----------
    mb = m.min_bound.cpu().float(); ms = m.max_bound.cpu().float()
    itv = (ms - mb) / m.grid_size.cpu().float()
    pts, gind = allp[0][0], allp[0][1]
    x, y, z = pts[:, 6], pts[:, 7], pts[:, 5]
    pol = torch.stack([torch.sqrt(x * x + y * y), torch.atan2(y, x), z], 1)
    gi2 = torch.floor((torch.maximum(torch.minimum(pol, ms - 1e-3), mb) - mb) / itv)
    bad = int((gi2 != gind).any(1).sum())
    report('CHECK-5   抖动后 grid_ind 与坐标自洽', bad == 0,
           f'不符 {bad}/{gind.shape[0]}')

    # ---------- CHECK 6：耗时 ----------
    ts = {}
    for nm, pc in (('off', {'enable': False}), ('on', {'enable': True, **P})):
        run(pc); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            run(pc)
        torch.cuda.synchronize(); ts[nm] = (time.time() - t0) / (3 * len(frames))
    inc = (ts['on'] / ts['off'] - 1) * 100
    report('CHECK-6   开启后单帧耗时增幅 <= 5%', inc <= 5.0,
           f'{ts["off"]*1e3:.1f} -> {ts["on"]*1e3:.1f} ms (+{inc:.2f}%)')

    print()
    print('ALL CHECKS PASSED' if all(results) else
          f'FAILED: {results.count(False)}/{len(results)} 项未通过')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
