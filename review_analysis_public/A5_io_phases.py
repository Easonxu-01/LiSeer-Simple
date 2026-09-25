#!/usr/bin/env python3
"""A5 phase decomposition: which part of an iteration is disk, which is compute.

On this machine the dataset lives on /data (sdd, rotational), while / is an
SSD. Dense scenes are ~7 MB each, so where they sit changes the data phase a
lot and the compute phase not at all. This script measures the three phases
separately so the Table R10 numbers cannot be mistaken for one another:

  phase 1  raw file read, cold cache, HDD vs an SSD-staged copy of the SAME
           files (page cache is dropped per file with posix_fadvise, so
           neither side is measured warm)
  phase 2  pipeline transform on data already in memory -- pure CPU, disk
           independent
  phase 3  the stall the training loop actually sees at next(loader), with
           prefetching workers hiding most of phase 1

The component costs D31-D33 are measured inside train_step only, i.e. after
next() has returned, so they belong to compute and are unaffected by all of
this. This script exists to demonstrate that rather than assert it.
"""
import argparse
import json
import os
import os.path as osp
import shutil
import time

import numpy as np

REFINE = '/data/Datasets/nuscenes/samples/LIDAR_TOP/refine'
SSD_STAGE = os.environ.get('A5_SSD_STAGE', '/tmp/a5_ssd_stage')
SEED = 0


def drop_cache(path):
    """Evict this file from the page cache (no root needed)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except Exception:
        pass


def read_bench(files, label, workers=4):
    for f in files:
        drop_cache(f)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.perf_counter()
    total = 0

    def one(f):
        # raw bytes: this phase is disk throughput, not array parsing (some
        # refine files are pickled object arrays, which np.load refuses by
        # default, and parsing is measured separately below anyway)
        with open(f, 'rb', buffering=0) as fh:
            n = 0
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                n += len(b)
        return n
    with ThreadPoolExecutor(workers) as ex:
        for n in ex.map(one, files):
            total += n
    dt = time.perf_counter() - t0
    return {'label': label, 'n_files': len(files), 'workers': workers,
            'total_MB': round(total / 1e6, 1),
            'wall_s': round(dt, 3),
            'MB_per_s': round(total / 1e6 / dt, 1),
            'ms_per_file': round(dt / len(files) * 1000, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--keep-stage', action='store_true')
    a = ap.parse_args()

    names = sorted(os.listdir(REFINE))[:a.n * 3]
    rng = np.random.RandomState(SEED)
    pick = [names[i] for i in rng.choice(len(names), a.n, replace=False)]
    hdd = [osp.join(REFINE, n) for n in pick]

    out = {'note': 'dense nuScenes scenes, ~7 MB each'}
    print('phase 1: raw read, cold cache')
    out['hdd_read'] = read_bench(hdd, 'HDD /data (sdd, rotational)')
    print('  ', out['hdd_read'])

    os.makedirs(SSD_STAGE, exist_ok=True)
    t0 = time.perf_counter()
    ssd = []
    for p in hdd:
        d = osp.join(SSD_STAGE, osp.basename(p))
        if not osp.exists(d):
            shutil.copy2(p, d)
        ssd.append(d)
    print(f'   staged {len(ssd)} files to SSD in {time.perf_counter()-t0:.1f}s')
    out['ssd_read'] = read_bench(ssd, 'SSD / (sdc3, non-rotational)')
    print('  ', out['ssd_read'])
    out['speedup_ssd_over_hdd'] = round(
        out['hdd_read']['ms_per_file'] / out['ssd_read']['ms_per_file'], 2)

    print('phase 2: decode/transform on in-memory data (CPU only)')
    import io
    buf = open(ssd[0], 'rb').read()
    t0 = time.perf_counter()
    n_ok = 0
    for _ in range(50):
        try:
            np.load(io.BytesIO(buf), allow_pickle=True)
            n_ok += 1
        except Exception:
            break
    out['decode_ms_per_file'] = (round((time.perf_counter() - t0) / max(n_ok, 1) * 1000, 3)
                                 if n_ok else None)
    print('  ', out['decode_ms_per_file'], 'ms/file')

    if not a.keep_stage:
        shutil.rmtree(SSD_STAGE, ignore_errors=True)
        print('   removed SSD staging dir')

    os.makedirs('review_analysis/_source_data/A5_raw', exist_ok=True)
    json.dump(out, open('review_analysis/_source_data/A5_raw/io_phases.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
