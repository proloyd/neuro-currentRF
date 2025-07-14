# Author: Proloy Das <email:proloyd94@gmail.com>
# License: BSD (3-clause)
"""Module implementing the FASTA algorithm"""

import numpy as np
from math import sqrt
import time

import logging


def _next_stepsize(deltax, deltaF):
    """A variation of spectral descent step-size selection: 'adaptive' BB method.

    Reference:
    ---------
    B. Zhou, L. Gao, and Y.H. Dai, 'Gradient methods with adaptive step-sizes,'
    Comput. Optim. Appl., vol. 35, pp. 69-86, Sept. 2006

    parameters
    ----------
    deltax: ndarray
        difference between coefs_current and coefs_next
    deltaF: ndarray
        difference between grad operator evaluated at coefs_current and coefs_next

    returns
    -------
    float
        adaptive step-size
    """
    n_deltax = (deltax ** 2).sum()  # linalg.norm(deltax, 'fro') ** 2
    n_deltaF = (deltaF ** 2).sum()  # linalg.norm(deltaF, 'fro') ** 2
    innerproduct_xF = np.real((deltax * deltaF).sum())

    if n_deltax == 0:
        return 0
    elif (n_deltaF == 0) | (innerproduct_xF == 0):
        return -1
    else:
        tau_s = n_deltax / innerproduct_xF  # steepest descent
        tau_m = innerproduct_xF / n_deltaF  # minimum residual

        # adaptive BB method
        if 2 * tau_m > tau_s:
            return tau_m
        else:
            return tau_s - 0.5 * tau_m


def _compute_residual(deltaf, sg):
    """Computes residuals"""
    res = sqrt(((deltaf + sg) ** 2).sum())
    a = sqrt((deltaf ** 2).sum())
    b = sqrt((sg ** 2).sum())
    res_r = res / (max(a, b) + 1e-15)
    return res, res_r


def _update_coefs(x, tau, gradfx, prox, f, g, beta, fk):
    """Non-monotone line search

    parameters
    ----------
    x: ndarray
        current coefficients
    tau: float
        step size
    gradfx: ndarry
        gradient operator evaluated at current coefficients
    prox: function handle
        proximal operator of :math:`g(x)`
    f: callable
        smooth differentiable function, :math:`f(x)`
    g: callable
        non-smooth function, :math:`g(x)`
    beta: float
        backtracking parameter
    fk: float
        maximum of previous function values

    returns
    -------
    z: ndarray
        next coefficients
    """
    x_hat = x - tau * gradfx
    z = prox(x_hat, tau)
    fz = f(z)
    count = 0
    while fz > fk + (gradfx * (z - x)).sum() + ((z - x) ** 2).sum() / (2 * tau):
        # np.square(linalg.norm(z - x, 'fro')) / (2 * tau):
        count += 1
        tau = beta * tau
        x_hat = x - tau * gradfx
        z = prox(x_hat, tau)
        fz = f(z)

    sg = (x_hat - z) / tau
    return z, fz, sg, tau, count


class Fasta:
    r"""Fast adaptive shrinkage/thresholding Algorithm

    Reference
    ---------
    Goldstein, Tom, Christoph Studer, and Richard Baraniuk. "A field guide to forward-backward
    splitting with a FASTA implementation." arXiv preprint arXiv:1411.3406 (2014).

    Parameters
    ----------
    f: function handle
        smooth differentiable function, :math:`f(x)`
    g: function handle
        non-smooth convex function, :math:`g(x)`
    gradf: function handle
        gradient of smooth differentiable function, :math:`\\nabla f(x)`
    proxg: function handle
        proximal operator of non-smooth convex function
     :math:`proxg(v, \\lambda) = argmin g(x) + \\frac{1}{2*\\lambda}\|x-v\|^2`
    beta: float, optional
        backtracking parameter
        default is 0.5
    n_iter: int, optional
        number of iterations
        default is 1000

    Attributes
    ----------
    coefs: ndvar
        learned coefficients
    objective_value: float
        optimum objective value
    residuals: list
        residual values at each iteration
    initial_stepsize: float, optional
        created only with verbose=1 option
    objective: list, optional
        objective values at each iteration
        created only with verbose=1 option
    stepsizes: list, optional
        stepsizes at each iteration
        created only with verbose=1 option
    backtracks: list, optional
        number of backtracking steps
        created only with verbose=1 option

    Notes
    -----
    Make sure that outputs of gradf and proxg is of same size as x.
    The implementation does not check for any such discrepancies.

    Use
    ---
    Solve following least square problem using fastapy
                :math:`\\min  .5||Ax-b||^2 + \\mu*\|x\|_1`
    Create function handles
    >>> def f(x): return 0.5 * linalg.norm(np.dot(A, x) - b, 2)**2  # f(x) = .5||Ax-b||^2
    >>> def gradf(x): return np.dot(A.T, np.dot(A, x) - b)  # gradient of f(x)
    >>> def g(x): return mu * linalg.norm(x, 1)  # mu|x|
    >>> def proxg(x, t): return shrink(x, mu*t)
    >>> def shrink(x, mu): return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0)) #proxg(z,t) = sign(x)*max(
    |x|-mu,0)
    Create FASTA instance
    >>> lsq = Fasta(f, g, gradf, proxg)
    Call solver
    >>> lsq.learn(x0, verbose=1)
    """

    def __init__(self, f, g, gradf, proxg, beta=0.5, n_iter=1000):
        self.f = f
        self.g = g
        self.grad = gradf
        self.prox = proxg
        self.beta = beta
        self.n_iter = n_iter
        self.residuals = []
        self._funcValues = []
        self.coefs_ = None

    def __str__(self):
        return "Fast adaptive shrinkage/thresholding Algorithm instance"

    def learn(self, coefs_init, tol=1e-2, verbose=True):
        """fits the model using FASTA algorithm

        parameters
        ----------
        coefs_init: ndarray
            initial guess

        tol: float, optional
            tolerance parameter
            default is 1e-8

        verbose: {0,1}
            verbosity of the method : 1 will display informations while 0 will display nothing
            default = 0

        returns
        -------
        self
        """
        logger = logging.getLogger("FASTA")
        coefs_current = np.copy(coefs_init)
        grad_current = self.grad(coefs_current)
        np.random.seed(0)  # seed the random generator to get same value
        coefs_next = coefs_current + 0.01 * np.random.randn(coefs_current.shape[0], coefs_current.shape[1])
        grad_next = self.grad(coefs_next)
        tau_current = _next_stepsize(coefs_next - coefs_current, grad_next - grad_current)

        self._funcValues.append(self.f(coefs_current))
        if verbose:
            self.objective = []
            self.objective.append(self._funcValues[-1] + self.g(coefs_current))
            self.initial_stepsize = np.copy(tau_current)
            self.stepsizes = []
            self.backtracks = []

        start = time.time()
        logger.debug("Iteration \t objective value \t step-size \t backtracking steps taken \t residual")
        for i in range(self.n_iter):
            coefs_next, objective_next, sub_grad, tau, n_backtracks = _update_coefs(coefs_current, tau_current,
                                                                                    grad_current, self.prox, self.f,
                                                                                    self.g, self.beta, max(self._funcValues))

            self._funcValues.append(objective_next)

            grad_next = self.grad(coefs_next)

            # Find residual
            delta_coef = coefs_current - coefs_next
            delta_grad = grad_current - grad_next
            residual, residual_r = _compute_residual(grad_next, sub_grad)
            self.residuals.append(residual)
            residual_n = residual / (self.residuals[0] + 1e-15)

            # Find step size for next iteration
            tau_next = _next_stepsize(delta_coef, delta_grad)

            if verbose:
                self.stepsizes.append(tau)
                self.backtracks.append(n_backtracks)
                self.objective.append(objective_next + self.g(coefs_next))
                logger.debug(f"{i} \t {self.objective[i]} \t {self.stepsizes[i]} \t {self.backtracks[i]} \t {self.residuals[i]}")

            # Prepare for next iteration
            coefs_current = coefs_next
            grad_current = grad_next

            if tau_next == 0 or min(residual_n, residual_r) < tol:  # convergence reached
                break
            elif tau_next < 0:  # non-convex probelms ->  negative stepsize -> use the previous value
                tau_current = tau
            else:
                tau_current = tau_next

        end = time.time()
        self.coefs_ = coefs_current
        self.objective_value = objective_next + self.g(coefs_current)
        if verbose:
            logger.debug(f"total time elapsed : {end - start}s")
