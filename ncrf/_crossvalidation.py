# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
from __future__ import annotations

import os
from math import ceil
from multiprocessing import Process, Queue
import queue
from typing import TYPE_CHECKING, List, Sequence

from eelbrain._config import CONFIG
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from ._model import NCRF, RegressionData


class CVResult:
    """Cross-validation results

    Parameters
    ----------
    mu
        Optimal ``mu`` parameter.
    weighted_l2_error
        self explanatory
    estimation_stability
        self explanatory
    cross_fit
        self explanatory
    l2_error
        L2 error from the optimal ``mu``.
    """

    def __init__(
            self, mu: float,
            weighted_l2_error: float,
            estimation_stability: float,
            cross_fit: float,
            l2_error: float,
    ):
        self.mu = mu
        self.weighted_l2_error = weighted_l2_error
        self.estimation_stability = 10 if np.isnan(estimation_stability) else estimation_stability  # replace Nan values with a big number
        self.cross_fit = cross_fit
        self.l2_error = l2_error


def naive_worker(fun, data, n_split, tol, job_q, result_q):
    """Worker function"""
    # myname = current_process().name
    if CONFIG['nice']:
        os.nice(CONFIG['nice'])
    while True:
        try:
            job = job_q.get_nowait()
            # print('%s got %s mus...' % (myname, len(job)))
            for mu in job:
                result_q.put(fun(data, n_split, tol, mu))
            # print('%s done' % myname)
        except queue.Empty:
            # print('returning from %s process' % myname)
            return


def start_workers(fun, data, n_split, tol, shared_job_q, shared_result_q, nprocs):
    """sets up workers"""
    procs = []
    for i in range(nprocs):
        p = Process(
            target=naive_worker,
            args=(fun, data, n_split, tol, shared_job_q, shared_result_q))
        procs.append(p)
        p.start()
    return procs


def crossvalidate(
        model: NCRF,
        data: RegressionData,
        mus: Sequence[float],
        tol: float,
        n_splits: int,
        n_workers: int = None,
) -> List[CVResult]:
    """used to perform cross-validation of cTRF model

    This function assumes `model` class has method _get_cvfunc(data, n_splits)
    which returns a callable. It calls that object with different
    regularizing weights (i.e. mus) to compute cross-validation metric and
    finally compares them to obtain the best weight.

    Parameters
    ----------
    model
        the model to be validated, here `NCRF`. In addition to that it needs to
        support the :func:`copy.copy` function.
    data
        Data.
    mus
        The range of the regularizing weights to test.
    tol
        Tolerance parameter. Decides when to stop outer iterations.
    n_splits
        number of folds for cross-validation.
    n_workers
        number of workers to be used. ``None`` to use ``cpu_count/2`` (default).

    Returns
    -------
    results : List[CVResult]
        Cross-validation results.
    """
    prog = tqdm(total=len(mus), desc="Crossvalidation", unit='mu', unit_scale=True)
    if n_workers is None:
        n = CONFIG['n_workers'] or 1  # by default this is cpu_count()
        n_workers = ceil(n / 8)

    fun = model.cvfunc

    job_q = Queue()
    result_q = Queue()

    for mu in mus:
        job_q.put([mu])  # put the job as a list.

    workers = start_workers(fun, data, n_splits, tol, job_q, result_q, n_workers)

    results = []
    for _ in range(len(mus)):
        result = result_q.get()
        results.append(result)
        prog.update(n=len(results))

    for worker in workers:
        worker.join()

    return results


class TimeSeriesSplit:
    def __init__(self, r=0.05, p=5, d=100):
        self.ratio = r
        self.p = p
        self.d = d

    def _iter_part_masks(self, X):
        n_v = ceil(self.ratio / (1 + self.ratio) * len(X))
        for i in range(self.p, 0, -1):
            test_mask = np.zeros(len(X), dtype=bool)
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[-(i * n_v + self.d):] = False
            if i == 1:
                test_mask[-i * n_v:] = True
            else:
                test_mask[-i * n_v:-(i - 1) * n_v] = True
            yield (train_mask, test_mask)

    def split(self, X):
        indices = np.arange(len(X))
        for (train_mask, test_mask) in self._iter_part_masks(X):
            train_index = indices[train_mask]
            test_index = indices[test_mask]
            yield train_index, test_index
