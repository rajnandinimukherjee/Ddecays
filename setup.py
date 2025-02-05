from typing import List, Dict, Tuple
import pdb
from scipy.optimize import least_squares
from scipy.special import gammaincc
import os
import typing
import h5py
from tqdm import tqdm
import pandas as pd
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_settings import plotparams
plt.rcParams.update(plotparams)

N_col = 3
dirs = ['X', 'Y', 'Z', 'T']
N_dir = len(dirs)


def st_dev(data, mean=None, **kwargs) -> np.ndarray:
    """standard deviation function - finds stdev
    around data mean or mean provided as input"""

    data, mean = data.real, mean.real
    n = len(data)
    if mean is None:
        mean = np.mean(data)
    return ((data - mean).dot(data - mean)/n)**0.5


class Stat:
    """ new datatype which stores bootstrap info and
    can interacts with scalars or other instances of
    the same class for basic mathematical operations """

    N_boot = 1000

    def __init__(self, val, err=None, btsp=None,
                 dtype=None, seed=None, **kwargs):
        self.val = np.array(val)
        self.shape = self.val.shape
        self.dtype = self.val.dtype if dtype is None else dtype

        accept_types = [np.ndarray, list, int, float, np.float64]
        self.err = np.array(err) if type(err) in accept_types else err
        self.btsp = np.array(btsp) if type(btsp) in accept_types else btsp

        if type(err) is str:
            if err == "fill":
                self.calc_err()
            elif err[-1] == "%":
                percent = float(err[:-1])
                dist = np.random.normal(
                    percent, percent / 4, size=self.val.shape)
                self.err = np.multiply(dist, self.val) / 100

        if type(btsp) is str:
            if btsp == "fill":
                self.calc_btsp()
            elif btsp == "seed":
                seed = kwargs["seed"]
                self.calc_btsp(seed=seed)
            elif btsp == "constant":
                self.btsp = np.repeat(
                    self.val[np.newaxis, ...], self.N_boot, axis=0)
        elif type(btsp) is np.ndarray and self.btsp.shape[0] != self.N_boot:
            self.N_boot = self.btsp.shape[0]

    def calc_err(self):
        if type(self.btsp) is np.ndarray:
            self.err = np.zeros(shape=self.shape)
            btsp = np.moveaxis(self.btsp, 0, -1)
            for idx, central in np.ndenumerate(self.val):
                self.err[idx] = st_dev(btsp[idx], central)
        else:
            self.err = np.zeros(shape=self.shape)

    def calc_btsp(self, seed=None):
        if type(self.err) is not np.ndarray:
            self.err = np.zeros(shape=self.shape)

        self.btsp = np.zeros(shape=self.shape + (self.N_boot,))
        for idx, central in np.ndenumerate(self.val):
            if type(seed) is not None:
                if type(seed) is str:
                    seed = int(hash(seed)) % (2**32)
                self.seed = seed
            else:
                if int(central) != 0:
                    digits = -int(np.log10(np.abs(central))) + 6
                    self.seed = np.abs(int(central * (10**digits)))
                else:
                    self.seed = 0

            np.random.seed(self.seed)
            self.btsp[idx] = np.random.normal(
                central, self.err[idx], self.N_boot)
        self.btsp = np.moveaxis(self.btsp, -1, 0)

    def use_func(self, func, **kwargs):
        central = func(self.val, **kwargs)

        btsp = np.array([func(self.btsp[k,], **kwargs)
                         for k in range(self.N_boot)])

        return Stat(val=central, err="fill", btsp=btsp)

    def __add__(self, other):
        if not isinstance(other, Stat):
            other = np.array(other)
            other = Stat(val=other, btsp="fill")
        new_stat = Stat(
            val=self.val + other.val,
            err="fill",
            btsp=np.array([self.btsp[k,] + other.btsp[k,]
                          for k in range(self.N_boot)]),
        )
        return new_stat

    def __sub__(self, other):
        if not isinstance(other, Stat):
            other = np.array(other)
            other = Stat(val=other, btsp="fill")
        new_stat = Stat(
            val=self.val - other.val,
            err="fill",
            btsp=np.array([self.btsp[k,] - other.btsp[k,]
                          for k in range(self.N_boot)]),
        )
        return new_stat

    def __mul__(self, other):
        if not isinstance(other, Stat):
            other = np.array(other)
            other = Stat(val=other, btsp="fill")
        new_stat = Stat(
            val=self.val * other.val,
            err="fill",
            btsp=np.array([self.btsp[k,] * other.btsp[k,]
                          for k in range(self.N_boot)]),
        )
        return new_stat

    def __matmul__(self, other):
        if not isinstance(other, Stat):
            other = np.array(other)
            other = Stat(val=other, btsp="fill")
        new_stat = Stat(
            val=self.val @ other.val,
            err="fill",
            btsp=np.array([self.btsp[k] @ other.btsp[k]
                          for k in range(self.N_boot)]),
        )
        return new_stat

    def __truediv__(self, other):
        if not isinstance(other, Stat):
            other = np.array(other)
            other = Stat(val=other, btsp="fill")
        new_stat = Stat(
            val=self.val / other.val,
            err="fill",
            btsp=np.array([self.btsp[k,] / other.btsp[k,]
                          for k in range(self.N_boot)]),
        )
        return new_stat

    def __pow__(self, num):
        new_stat = Stat(
            val=self.val**num,
            err="fill",
            btsp=np.array([self.btsp[k,] ** num for k in range(self.N_boot)]),
        )
        return new_stat

    def __neg__(self):
        new_stat = Stat(val=-self.val, err=self.err, btsp=-self.btsp)
        return new_stat

    def __getitem__(self, indices):
        key = indices
        if not isinstance(key, tuple):
            key = (key,)
        new_stat = Stat(val=self.val[key],
                        err=self.err[key],
                        btsp=self.btsp[(slice(None),)+key])
        return new_stat


Zero = Stat(val=0, err=0, btsp='fill')


def join_stats(stats):
    return Stat(
        val=np.array([s.val for s in stats]),
        err=np.array([s.err for s in stats]),
        btsp=np.array([s.btsp for s in stats]).swapaxes(0, 1),
    )


def bootstrap(data: np.ndarray, Nboot: int = Stat.N_boot, seed=1) -> np.ndarray:
    """bootstrap samples generator -
    if input data has same size as K,
    assumes it's already a bootstrap sample
    and does no further sampling"""

    if type(seed) is str:
        seed = int(hash(seed)) % (2**32)

    C = data.shape[0]
    if C == Nboot:  # goes off when data itself is bootstrap data
        samples = data
    else:
        np.random.seed(seed)
        slicing = np.random.randint(0, C, size=(C, Nboot))
        samples = np.mean(
            data[
                tuple(slicing.T if ax == 0 else slice(None)
                      for ax in range(data.ndim))
            ],
            axis=1,
        )
    return np.array(samples, dtype=data.dtype)


def callPDF(filename: str, show: bool = True) -> None:
    """ plots matplotlib graphics into pdfs, saves and shows"""

    pdf = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format="pdf")
    pdf.close()
    plt.close("all")
    if show:
        os.system("open " + filename)


def err_disp(num, err, n=2, sys_err=None, **kwargs):
    """ converts num and err into num(err) in
    scientific notation upto n digits in error """

    if err == 0.0:
        return str(np.around(num, 2))
    else:
        if type(sys_err) is not None:
            err_size = max(
                int(np.floor(np.log10(np.abs(err)))),
                int(np.floor(np.log10(np.abs(sys_err)))),
            )
        else:
            err_size = int(np.floor(np.log10(np.abs(err))))

        num_size = int(np.floor(np.log10(np.abs(num))))
        min_size = min(err_size, num_size + (n - 1))
        err_n_digits = int(np.round(err * 10 ** (-(min_size - (n - 1)))))

        if min_size > (n - 1):
            disp_str = f"{num}({err})"
        else:
            disp_str = "{:.{m}f}".format(num, m=-(min_size - (n - 1)))
            disp_str += f"({err_n_digits})"

        if type(sys_err) is not None:
            sys_err_n_digits = int(
                np.round(sys_err * 10 ** (-(min_size - (n - 1)))))
            disp_str += f"({sys_err_n_digits})"

        return disp_str


def COV(data, **kwargs):
    """covariance matrix calculator - accounts for cov matrices
    centered around sample avg vs data avg and accordingly normalises"""

    C, T = data.shape

    if "center" in kwargs.keys():
        center = kwargs["center"]
        norm = C
    else:
        center = np.mean(data, axis=0)
        norm = C - 1

    COV = np.array([[((data[:, t1] - center[t1]).
                    dot(data[:, t2] - center[t2])) / norm
                   for t2 in range(T)]for t1 in range(T)])

    return COV


def m_eff(data, ansatz="cosh", **kwargs):
    if ansatz == "cosh":
        m_eff = np.arccosh(0.5 * (data[2:] + data[:-2]) / data[1:-1])
    elif ansatz == "exp":
        m_eff = np.abs(np.log(data[1:] / data[:-1]))
    return m_eff.real


def fit_func(
    x,
    y,
    ansatz,
    guess,
    start=0,
    end=None,
    verbose=False,
    correlated=False,
    pause=False,
    chi_sq_rescale=False,
    Nboot=100,
    **kwargs,
):

    if not isinstance(x, Stat):
        x = Stat(
            val=x,
            err=np.zeros(shape=np.array(x).shape),
            btsp='fill')

    if type(end) is None:
        end = len(x.val)

    if correlated:
        cov = COV(y.btsp[:, start:end], center=y.val[start:end])
        L_inv = np.linalg.cholesky(cov)
        L = np.linalg.inv(L_inv)
    else:
        L = np.diag(1 / y.err[start:end])

    def diff(inp, out, param, **kwargs):
        return out - ansatz(inp, param, **kwargs)

    def LD(param):
        return L.dot(diff(x.val[start:end], y.val[start:end],
                          param, fit="central"))

    res = least_squares(LD, guess, ftol=1e-10, gtol=1e-10)
    if verbose:
        print(res)

    chi_sq = LD(res.x).dot(LD(res.x))
    DOF = len(x.val[start:end]) - np.count_nonzero(guess)
    pvalue = gammaincc(DOF / 2, chi_sq / 2)

    res_btsp = np.zeros(shape=(Nboot, len(guess)))
    for k in range(Nboot):

        def LD_k(param):
            return L.dot(
                diff(x.btsp[k, start:end], y.btsp[k, start:end],
                     param, fit="btsp", k=k)
            )

        res_k = least_squares(LD_k, guess, ftol=1e-10, gtol=1e-10)
        res_btsp[k,] = res_k.x

    res = Stat(val=res.x, err="fill", btsp=res_btsp)
    if pvalue < 0.05 and chi_sq_rescale:
        res = Stat(val=res.val, err=res.err *
                   ((chi_sq / DOF) ** 0.2), btsp="fill")

    def mapping(xvals):
        if not isinstance(xvals, Stat):
            xvals = Stat(val=np.array(xvals), btsp="fill")

        return Stat(
            val=ansatz(xvals.val, res.val, fit="recon"),
            err="fill",
            btsp=np.array(
                [ansatz(xvals.btsp[k,], res.btsp[k], fit="recon")
                 for k in range(Nboot)]
            ),
        )

    res.mapping = mapping
    res.chi_sq = chi_sq
    res.DOF = DOF
    res.pvalue = pvalue
    res.range = (start, end)
    if pause:
        pdb.set_trace()

    return res


def foldcorr(corr: np.ndarray, T: int) -> np.ndarray:
    return 0.5*(corr[1:]+corr[::-1][:-1])[:int(T/2)]
