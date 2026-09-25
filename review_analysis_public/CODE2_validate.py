"""CODE-2 (DensePointDegrade) patch self-check.

    python review_analysis/CODE2_validate.py
"""
import sys, os
sys.path.insert(0, os.getcwd())
import torch


def main():
    import projects.unilidar_plugin.datasets.pipelines as P
    from mmdet.datasets.builder import PIPELINES

    ok1 = hasattr(P, 'DensePointDegrade')
    print(f'[1/4] DensePointDegrade importable from pipelines package: '
          f'{"OK" if ok1 else "FAIL"}')

    ok2 = 'DensePointDegrade' in PIPELINES.module_dict
    print(f'[2/4] DensePointDegrade registered in PIPELINES registry: '
          f'{"OK" if ok2 else "FAIL"}  '
          f'(if this is False but [1/4] is True, you imported the class but '
          f'not the module that runs the @PIPELINES.register_module() '
          f'decorator -- re-check the __init__.py edit)')

    from projects.unilidar_plugin.datasets.pipelines.loading_voxels_sk import DensePointDegrade

    # [3/4] no-op when dense7sparse is not set (e.g. reset_random / pure sparse samples)
    tf = DensePointDegrade(sigma=0.06)
    pts_before = torch.randn(100, 8)
    results = {'points': pts_before.clone()}
    out = tf(results)
    ok3 = torch.equal(out['points'], pts_before)
    print(f'[3/4] no-op without dense7sparse=True: {"OK" if ok3 else "FAIL"}')

    # [4/4] with dense7sparse=True, only the dense tail moves, sparse head is untouched
    n_sparse, n_dense = 40, 60
    pts_before = torch.randn(n_sparse + n_dense, 8)
    torch.manual_seed(0)
    results = {'points': pts_before.clone(), 'dense7sparse': True,
               'num_sparse_points': n_sparse}
    out = tf(results)
    sparse_unchanged = torch.equal(out['points'][:n_sparse], pts_before[:n_sparse])
    dense_moved = not torch.equal(out['points'][n_sparse:, :3], pts_before[n_sparse:, :3])
    dense_other_dims_unchanged = torch.equal(
        out['points'][n_sparse:, 3:], pts_before[n_sparse:, 3:])
    shift = (out['points'][n_sparse:, :3] - pts_before[n_sparse:, :3])
    plausible_scale = float(shift.std()) < 0.06 * 5  # sigma=0.06, generous bound
    ok4 = sparse_unchanged and dense_moved and dense_other_dims_unchanged and plausible_scale
    print(f'[4/4] sparse half untouched, dense xyz perturbed '
          f'(shift std={float(shift.std()):.4f}, sigma=0.06), other dims untouched: '
          f'{"OK" if ok4 else "FAIL"}')

    if not (ok1 and ok2 and ok3 and ok4):
        raise SystemExit('CODE-2 patch failed validation')
    print('\nALL CHECKS PASSED.')


if __name__ == '__main__':
    main()
