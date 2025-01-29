from setup import *
from ensemble_parameters import parameters as pars


def _get_file_struct(path: str) -> Dict:
    folders = os.listdir(path)
    all_data
    return folders


class Ensemble:
    path = '/home/dp207/dp207/shared/projects/hadronic_D_decays/NPR/'
    N_cf = 10
    cf_list = np.arange(100, 1000+1, 100)

    def __init__(self, name: str) -> None:
        self.name = name
        self.L = pars[name]['L']
        self.T = pars[name]['T']
        self.a = pars[name]['a']
        self.ainv = 1/self.a

        self.seed = int(hash(name)) % (2**32)
        self.datafolder = self.path+'t{}l{}_b{}_k{}{}_csw{}/npr_data'.format(
            self.T,
            self.L,
            pars[name]["beta"],
            pars[name]["kappa_add"],
            pars[name]["kappa"],
            pars[name]["csw"]
        )
