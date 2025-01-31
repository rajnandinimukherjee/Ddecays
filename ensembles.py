from setup import *
from ensemble_parameters import parameters as pars


def _get_file_struct(path: str) -> Dict:
    folders = os.listdir(path)
    all_data
    return folders


class Ensemble:
    path = '/home/dp207/dp207/shared/projects/hadronic_D_decays/NPR/'
    cf_list = np.arange(100, 1000+1, 100)

    def __init__(self, name: str) -> None:
        self.name = name
        self.L = pars[name]['L']
        self.T = pars[name]['T']
        self.a = pars[name]['a']
        self.ainv = 1/self.a

        self.seed = int(hash(name)) % (2**32)
        self.datafolder = self.path+'t{}l{}_b{}_k{}{}_csw{}'.format(
            self.T,
            self.L,
            pars[name]["beta"],
            pars[name]["kappa_add"],
            pars[name]["kappa"],
            pars[name]["csw"]
        )

    def config_counter(self, prefix: str = 'two_point', data: str = 'valence') -> None:

        cfgs = {}

        name = self.datafolder.rsplit('/')[-1]
        if data == 'valence':
            path = self.path+f'hadronic_ward_identity/{name}/s0g0'
            mass_map = {mass_str2float('m'+mass.rsplit('_')[0]): 'm'+mass.rsplit('_')[0]
                        for mass in os.listdir(path)}
        elif data == 'NPR':
            path = self.path+f'{name}/npr_data'
            mass_map = {mass_str2float(
                mass): mass for mass in os.listdir(path)}
        else:
            print('data is either valence or NPR (str)')

        masses = sorted(list(mass_map.keys()))
        for mass in masses:
            if data == 'valence':
                folder = f'{path}/{mass_map[mass][1:]}_s0g0/mesons/'
            else:
                folder = f'{path}/{mass_map[mass]}/p2_2p/'

            vals = sorted(list(set([f.rsplit('.')[-2] for f in os.listdir(folder)
                                    if f.startswith(prefix)])))

            cfgs[str(np.around(-mass, 3))] = {
                'N_cf': len(vals),
                'cfgs': f'{vals[0]}->{vals[-1]}'
            }

        df = pd.DataFrame.from_dict(cfgs, orient='columns')
        print(df.to_string())


def mass_str2float(mass: str) -> float:
    mass = re.search(r'p(\d+\.\d+|\d+)', mass)
    return float(f"0.{mass.group(1)}")
