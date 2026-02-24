import os
import sys
import time

import pytest

from chemprop.utils import make_mol, parallel_execute


@pytest.mark.skipif(
    sys.platform in ["win32", "darwin"], reason="Multiprocessing can hang on Windows and MacOS."
)
def test_parallel_execution():
    def add_two(x, y):
        return x + y

    expected_result = [3, 7]

    results_single_worker = parallel_execute(add_two, [[1, 2], [3, 4]], n_workers=1)
    assert results_single_worker == expected_result

    results_multi_worker = parallel_execute(add_two, [[1, 2], [3, 4]], n_workers=2)
    assert results_multi_worker == expected_result


# @pytest.mark.skip(reason="Debuggers can slow down multiprocessing.")
@pytest.mark.skipif(
    os.cpu_count() < 4, reason="Speedup is expected if multiple threads are available."
)
@pytest.mark.skipif(
    sys.platform in ["win32", "darwin"], reason="Multiprocessing can hang on Windows and MacOS."
)
def test_parallel_is_faster():
    smis = ["C1=CC=C(N=C1)C1=CC=C(N=C1)C1=CC=C(N=C1)C1=CC=C(Cl)N=C1" * 100] * 4
    keep_h, add_h, ignore_stereo, reorder_atoms = False, False, False, False

    start_time = time.time()
    parallel_results = parallel_execute(
        make_mol, [(smi, keep_h, add_h, ignore_stereo, reorder_atoms) for smi in smis], n_workers=4
    )
    parallel_runtime = time.time() - start_time

    start_time = time.time()
    sequential_results = [
        make_mol(smi, keep_h, add_h, ignore_stereo, reorder_atoms) for smi in smis
    ]
    sequential_runtime = time.time() - start_time

    assert len(sequential_results) == len(parallel_results)
    assert parallel_runtime < sequential_runtime
