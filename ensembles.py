from setup import *
from ensemble_parameters import parameters as pars


class Ensemble:
    path = '/home/dp207/dp207/shared/projects/hadronic_D_decays/NPR'

    def __init__(self, name: str) -> None:
        self.name = name
        self.L = pars[name]['L']
        self.T = pars[name]['T']
        self.a = pars[name]['a']
        self.ainv = 1/self.a

        self.seed = int(hash(name)) % (2**32)
        self.dataname = 't{}l{}_b{}_k{}{}_csw{}'.format(
            self.T,
            self.L,
            pars[name]["beta"],
            pars[name]["kappa_add"],
            pars[name]["kappa"],
            pars[name]["csw"]
        )

    def __repr__(self):
        return self.dataname

    def config_counter(self, data: str, prefix: str,
                       show: bool = True) -> Tuple[Dict, List]:

        cfgs = {}

        if data == 'valence':
            path = f'{self.path}/hadronic_ward_identity/{self.dataname}/s0g0'
        elif data == 'NPR':
            path = f'{self.path}/new_runs/{self.dataname}/npr_data'
            # path = f'{self.path}/{self.dataname}/npr_data'
        else:
            print('data is either valence or NPR')

        mass_map = {mass_str2float(mass): mass
                    for mass in os.listdir(path)}
        masses = sorted(list(mass_map.keys()))
        for mass in masses:
            if data == 'valence':
                folder = f'{path}/{mass_map[mass]}/mesons/'
            else:
                momenta = os.listdir(f'{path}/{mass_map[mass]}/')
                folder = f'{path}/{mass_map[mass]}/{momenta[0]}'

            vals = sorted(map(int, list(set([f.rsplit('.')[-2] for f in os.listdir(folder)
                                            if f.startswith(prefix)]))))

            try:
                cfgs[str(np.around(mass, 3))] = {
                    'N_cf': len(vals),
                    'cfgs': f'{vals[0]}->{vals[-1]}'
                }
            except IndexError:
                pdb.set_trace()

            if data == 'NPR' and show:
                momvars = np.mean([len([f for f in os.listdir(f'{path}/{mass_map[mass]}/{momenta[i]}')
                                        if f.startswith(prefix) and f.endswith(str(vals[-1])+'.h5')])
                                   for i in range(len(momenta))])
                cfgs[str(np.around(mass, 3))]['N_tw'] = momvars

        if show:
            df = pd.DataFrame.from_dict(cfgs, orient='columns')
            print(df.to_string())
        else:
            return mass_map, vals


def mass_str2float(mass: str) -> float:
    mass = mass.rsplit('_')[0]
    mass = mass.replace('m', '').replace('p', '.').replace('n', '-')
    return float(mass)


def convert_to_phys(vec: np.ndarray, L: int, T: int) -> np.ndarray:
    vec = np.array(list(map(float, vec)))
    L, T = L/(2*np.pi), T/(2*np.pi)
    return np.array(list(vec[:3]/L)+[vec[-1]/T])


def decode_fname(fname: str) -> Tuple[int, List, List]:
    components = fname.rsplit('_')
    mom1 = components[3:7]
    mom2 = components[7:10]
    cfg = components[-1].rsplit('.')[-2]
    num = components[-1].rsplit('.'+cfg)[0]

    mom2.append(num)
    return int(cfg), mom1, mom2


def SMOM_combo_sort(arr: np.ndarray) -> np.ndarray:
    """ sorts momentum combinations in the form
        [[A,A,0,0], [B,0,B,0],
         [A,A,A,A], [0,0,0,B],
         [A,A,A,A], [B,B,B,B]]
    """
    new_arr = np.empty(shape=arr.shape, dtype=arr.dtype)
    for i in range(3):
        A, B = arr[i, 0, :], arr[i, 1, :]
        if np.all(A != '0.0'):
            if np.all(B[:3] == '0.0'):
                idx = 1
            else:
                idx = 2
        else:
            idx = 0
        new_arr[idx, 0], new_arr[idx, 1] = A, B

    return new_arr


SMOM_combos = [
    r'$\left(p_1^x-p_2^x,p_1^y,-p_2^z,0\right)$',
    r'$\left(p_1^x,p_1^y,p_1^z,p_1^t-p_2^t\right)$',
    r'$\left(p_1^x-p_2^x,p_1^y-p_2^y,p_1^z-p_2^z,p_1^t-p_2^t\right)$'
]


def MOM_combo_sort(arr: np.ndarray) -> np.ndarray:
    """ sorts momentum combinations in the form
    [[0,0,0,A],  [0,0,0,B],
     [A,0,0,A],  [B,0,0,B],
     [A,0,A,0],  [B,0,B,0],
     [A,A,A,A],  [B,B,B,B],
     [A,A,A,-A], [B,B,B,-B]]
    """
    new_arr = np.empty(shape=arr.shape, dtype=arr.dtype)
    for i in range(5):
        A, B = arr[i, 0, :], arr[i, 1, :]
        if np.all(A != '0.0'):
            if A[-1][0] == '-':
                idx = 4
            else:
                idx = 3
        elif np.all(A[:3] == '0.0'):
            idx = 0
        else:
            if np.all(A[1:3] == '0.0') or np.all(A[2:] == '0.0'):
                idx = 1
            else:
                idx = 2
        new_arr[idx, 0], new_arr[idx, 1] = A, B

    return new_arr


MOM_combos = [
    r'0_0_0_A__0_0_0_B',
    r'A_0_0_A__B_0_0_B',
    r'A_0_A_0__B_0_B_0',
    r'A_A_A_A__B_B_B_B',
    r'A_A_A_-A__B_B_B_-B',
]

MOM_combos = [
    r'$\left(0,0,0,p_1^t-p_2^t\right)$',
    r'$\left(p_1^x-p_2^x,0,0,p_1^t-p_2^t\right)$',
    r'$\left(p_1^x-p_2^x,0,p_1^z-p_2^z,0\right)$',
    r'$\left(p_1^x-p_2^x,p_1^y-p_2^y,p_1^z-p_2^z,p_1^t-p_2^t\right)$',
    r'$\left(p_1^x-p_2^x,p_1^y-p_2^y,p_1^z-p_2^z,-p_1^t+p_2^t\right)$',
]
