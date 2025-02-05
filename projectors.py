from setup import *

N_cd = N_col*N_dir

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


def g5(prop: np.ndarray) -> np.ndarray:
    gamma5 = Gamma['5'].swapaxes(1, 2).reshape((N_cd, N_cd), order='F')
    return gamma5@prop.conj().T@gamma5


def bilinear_projectors(subscheme: str, qvec: np.ndarray) -> Dict:
    if subscheme not in ['gamma', 'qslash']:
        raise 'subscheme input is either gamma or qslash (str)'

    myGamma = {label: mtx.swapaxes(1, 2).reshape((N_cd, N_cd), order='F')
               for label, mtx in Gamma.items()}

    if subscheme == 'qslash':
        qslash = np.sum([qvec[i] * myGamma[dirs[i]]
                         for i in range(N_dir)], axis=0)
        qsq = qvec.dot(qvec)
        # replace \gamma_\mu with \slashed{q}q_\mu/q^2
        for i in range(N_dir):
            myGamma[dirs[i]] = qslash*qvec[i]/qsq

    projectors = {
        "S": [myGamma["I"]],
        "P": [myGamma["5"]],
        "V": [myGamma[i] for i in dirs],
        "A": [myGamma[i]@myGamma["5"] for i in dirs],
        "T": sum([[myGamma[dirs[i]]@myGamma[dirs[j]]
                   for j in range(i+1, N_dir)]
                  for i in range(0, N_dir-1)], [],)}

    tree_values = {curr: np.sum([np.trace(mtx@mtx) for mtx in proj], axis=0)
                   for curr, proj in projectors.items()}

    return {curr: [Stat(val=mtx/tree_values[curr], btsp='constant') for mtx in proj]
            for curr, proj in projectors.items()}


def fourquark_projectors(subscheme: str, qvec: np.ndarray) -> Dict:
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

    doubles = {
        "AV": [np.einsum('abcd,efgh->abcdefgh')]
    }
