from setup import *

gamma = {
    "I": np.identity(N_dir, dtype="complex128"),
    "X": np.zeros(shape=(N_dir, N_dir), dtype="complex128"),
    "Y": np.zeros(shape=(N_dir, N_dir), dtype="complex128"),
    "Z": np.zeros(shape=(N_dir, N_dir), dtype="complex128"),
    "T": np.zeros(shape=(N_dir, N_dir), dtype="complex128"),
}

for i in range(N_dir):
    gamma["X"][i, N_dir - i - 1] = 1j if i <= 1 else -1j
    gamma["Y"][i, N_dir - i - 1] = 1 if (i == 1 or i == 2) else -1
    gamma["Z"][i, (i + 2) % N_dir] = (-1j) if (i == 1 or i == 2) else 1j
    gamma["T"][i, (i + 2) % N_dir] = 1

gamma["5"] = gamma["X"] @ gamma["Y"] @ gamma["Z"] @ gamma["T"]

# =====put color structure into gamma matrices====================
Gamma = {
    name: np.einsum("ab,cd->abcd", mtx, np.identity(N_col))
    for name, mtx in gamma.items()
}


def tensordot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """ matrix multiplication of (N_dir,N_dir,N_col,N_col) data
    to contract spacetime and colour indices; equivalent to
    np.einsum('abcd,bedf->aecf', A, B)
    """
    return np.tensordot(A, B, axes=([1, 3], [0, 2])).transpose(0, 2, 1, 3)


def Stattensordot(A, B) -> Stat:
    """ tensordot method under the bootstrap """
    if type(A) is not Stat:
        A = Stat(
            val=A,
            err=np.zeros(shape=A.shape),
            btsp='fill'
        )
    if type(B) is not Stat:
        B = Stat(
            val=B,
            err=np.zeros(shape=B.shape),
            btsp='fill'
        )
    return Stat(
        val=tensordot(A.val, B.val),
        err='fill',
            btsp=np.array([tensordot(A.btsp[k], B.btsp[k])
                           for k in range(A.btsp.shape[0])])
    )


def tensortrace(A: np.ndarray) -> np.ndarray:
    """ trace of (N_dir,N_dir,N_col,N_col) data
    over spacetime and colour indices
    """
    return np.trace(A.swapaxes(1, 2).reshape(12, 12))


def tensorinv(A: np.ndarray) -> np.ndarray:
    """ inverse of (N_dir,N_dir,N_col,N_col) data """
    return np.linalg.inv(A.swapaxes(1, 2).reshape((12, 12), order='F')).reshape(
        (4, 3, 4, 3), order='F').swapaxes(2, 1)


def tensorhermitian(A: np.ndarray) -> np.ndarray:
    """ hermitian conjugate of (N_dir,N_dir,N_col,N_col) data """
    return (np.conj(A.swapaxes(1, 2).reshape((12, 12), order='F')).T).reshape(
        (4, 3, 4, 3), order='F').swapaxes(2, 1)


def G5H(A: np.ndarray) -> np.ndarray:
    """ gamma5 hermitian conj of (N_dir,N_dir,N_col,N_col) data """
    return tensordot(Gamma['5'], tensordot(tensorhermitian(A), Gamma['5']))


def bilinear_projectors(subscheme: str, qvec: np.ndarray) -> Dict:
    if subscheme == 'gamma':
        sGamma = {i: Gamma[i] for i in dirs}

    elif subscheme == 'qslash':
        if type(qvec) is not np.ndarray:
            raise 'Need vector q (np.ndarray) for qslash scheme'
        qslash = np.sum([qvec[i] * Gamma[dirs[i]]
                         for i in range(N_dir)], axis=0)
        qsq = qvec.dot(qvec)

        # replace \gamma_\mu with \slashed{q}q_\mu/q^2
        sGamma = {dirs[i]: qslash*qvec[i]/qsq for i in range(N_dir)}

    else:
        raise 'subscheme input is either gamma or qslash (str)'

    projectors = {
        "S": [Gamma["I"]],
        "P": [Gamma["5"]],
        "V": [sGamma[i] for i in dirs],
        "A": [tensordot(sGamma[i], Gamma["5"]) for i in dirs],
        "T": sum([[tensordot(sGamma[dirs[i]], sGamma[dirs[j]])
                   for j in range(i+1, N_dir)]
                  for i in range(0, N_dir-1)], [],)}
    tree_values = {curr: np.sum([tensortrace(tensordot(mtx, mtx))
                                 for mtx in proj], axis=0)
                   for curr, proj in projectors.items()}
    return {curr: [mtx/tree_values[curr] for mtx in proj]
            for curr, proj in projectors.items()}
