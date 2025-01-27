from setup import *
from ensemble_parameters import parameters as pars


def _get_masses(path: str) -> List:
    return [float('0.{}'.format(mass[4:])) for mass in os.listdir(path)]


class Ensemble:
    path = '/home/dp207/dp207/shared/projects/hadronic_D_decays/NPR/'

    def __init__(self, name: str) -> None:
        self.name = name
        self.L = pars[name]['L']
        self.T = pars[name]['T']
        self.seed = int(hash(name)) % (2**32)
        self.datafolder = self.path+'t{}l{}_b{}_k{}{}_csw{}/npr_data/'.format(
            self.T,
            self.L,
            pars[name]["beta"],
            pars[name]["kappa_add"],
            pars[name]["kappa"],
            pars[name]["csw"]
        )
        self.masses = _get_masses(self.datafolder)
