from ensembles import *

class Bilinears:

    def __init__(self, ensemble:str, scheme:str='SMOM') -> None:
        self.ens = Ensemble(ensemble)
        self.scheme = scheme
        self.prefix = f'{scheme}_Bilinear_'
        self.masses = self.ens.masses
        self.all_files = {mass:os.listdir('{}/{}/'.format(self.ens.datafolder,path) for mass, path in zip(self.ens.masses, self.ens.mass_paths)}

