from setup import *
from ensemble_parameters import parameters as pars


class Ensemble:
    path = '/home/dp207/dp207/shared/projects/hadronic_D_decays/NPR/'

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
        self.datafolder = self.path+'new_runs/'+self.dataname

    def __repr__(self):
        return self.dataname

    def config_counter(self, data: str, prefix: str,
                       show: bool = True) -> Tuple[Dict, List]:

        cfgs = {}

        name = self.datafolder.rsplit('/')[-1]
        if data == 'valence':
            path = self.path+f'hadronic_ward_identity/{name}/s0g0'
        elif data == 'NPR':
            path = self.datafolder+'/npr_data'
        else:
            print('data is either valence or NPR')

        mass_map = {mass_str2float(mass): mass
                    for mass in os.listdir(path)}
        masses = sorted(list(mass_map.keys()))
        for mass in masses:
            if data == 'valence':
                folder = f'{path}/{mass_map[mass]}/mesons/'
            else:
                mom_folder = os.listdir(f'{path}/{mass_map[mass]}/')[0]
                folder = f'{path}/{mass_map[mass]}/{mom_folder}/'

            vals = sorted(map(int, list(set([f.rsplit('.')[-2] for f in os.listdir(folder)
                                            if f.startswith(prefix)]))))

            try:
                cfgs[str(np.around(mass, 3))] = {
                    'N_cf': len(vals),
                    'cfgs': f'{vals[0]}->{vals[-1]}'
                }
            except IndexError:
                pdb.set_trace()

        if show:
            df = pd.DataFrame.from_dict(cfgs, orient='columns')
            print(df.to_string())
        else:
            return mass_map, vals


def mass_str2float(mass: str) -> float:
    mass = mass.rsplit('_')[0]
    mass = mass.replace('m', '-').replace('p', '.').replace('n', '')
    return float(mass)
