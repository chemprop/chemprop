from chemprop.utils import parallel_execute


def test_parallel_execution():
    def add_two(x, y):
        return x + y

    expected_result = [3, 7]

    results_single_worker = parallel_execute(add_two, [[1, 2], [3, 4]], n_workers=1)
    assert results_single_worker == expected_result

    results_multi_worker = parallel_execute(add_two, [[1, 2], [3, 4]], n_workers=2)
    assert results_multi_worker == expected_result

    results_zipped_multi_worker = parallel_execute(
        add_two, [[1, 3], [2, 4]], n_workers=2, zipped=False
    )
    assert results_zipped_multi_worker == expected_result
