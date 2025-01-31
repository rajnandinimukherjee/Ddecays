from projectors import *
from pion_mass import *


class Bilinear:
    currents = ['S', 'P', 'V', 'A', 'T']

    def __init__(self, ensemble: str, scheme: str = 'SMOM',
                 compute: bool = False) -> None:
        self.ens = Ensemble(ensemble)
        self.scheme = scheme
        self.prefix = f'{scheme}_Bilinear_00_'
        self.path = self.ens.datafolder+'/npr_data'
        self.compute = compute
        self.cf_list = pars[self.ens.name]['NPR_cfgs']
        self.N_cf = len(self.cf_list)

        name = self.path.rsplit('/')[-2]
        self.Zdata_fname = f'Z_factors/{name}.hd5'

        if self.compute:
            self.create_attributes()

    def chiral_extrap(self, momvar_idx: int, subscheme: str,
                      plot: bool = False) -> Dict:
        Zs = self.get_all_Zs(momvar_idx, subscheme)
        extrap = {}

        for c_idx, current in enumerate(self.currents):
            extrap[current] = []
            for m_idx, mom in enumerate(self.momenta):
                ys = join_stats([Zs[mass][current][m_idx]
                                 for mass in self.masses])
                res = fit_func(self.masses, ys, chiral_ansatz,
                               [1, 1], correlated=True)
                extrap[current].append(res[0])
            extrap[current] = join_stats(extrap[current])

        if plot:
            sublabel = r'\gamma_\mu' if subscheme == 'gamma' else r'\not{q}'
            title = self.scheme+r'$^{'+sublabel + \
                r'}$, $m_\pi=0$ mom combo '+str(momvar_idx+1)
            fname = f'plots/{self.ens.name}_Zs_chiral_extrap_tw{momvar_idx}.pdf'
            self.plot_Z_bls(extrap, title, fname)

        return extrap

    def get_all_Zs(self, momvar_idx: int, subscheme: str) -> Dict:
        if not self.compute:
            mass_str = list(h5py.File(self.Zdata_fname, 'r')
                            [f'{subscheme}/Bilinear'].keys())
            self.mass_map = {mass_str2float(mass): mass for mass in mass_str}
            self.masses = sorted(list(self.mass_map.keys()))

        Zs = {mass: self.get_Zs(mass, momvar_idx, subscheme)
              for mass in self.masses}

        fig, ax = plt.subplots(nrows=len(self.currents),
                               ncols=1, figsize=(3, 10))
        plt.subplots_adjust(hspace=0)
        sublabel = r'\gamma_\mu' if subscheme == 'gamma' else r'\not{q}'
        title = self.scheme+r'$^{'+sublabel + \
            r'}$, mom combo '+str(momvar_idx+1)
        ax[0].set_title(title)

        for c_idx, current in enumerate(self.currents):
            for m_idx, mass in enumerate(self.masses):
                ax[c_idx].errorbar(self.momenta, Zs[mass][current].val,
                                   yerr=Zs[mass][current].err, fmt='o',
                                   capsize=4, label=f'{np.around(mass, 3)}')
            ax[c_idx].set_ylabel(r'$Z_'+current+r'/Z_q$')

        ax[-1].set_xlabel(r'$\sqrt{q^2}$ [GeV]')
        handles, labels = ax[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

        fname = f'plots/{self.ens.name}_Zs_all_masses_tw{momvar_idx}.pdf'
        callPDF(fname, show=False)
        print(f'plotted to {os.getcwd()}/{fname}')

        return Zs

    def get_Zs(self, mass: float, momvar_idx: int,
               subscheme: str, plot: bool = False) -> Dict:

        Zs = {}

        if self.compute:
            for idx in tqdm(range(len(self.momenta)),
                            leave=False, desc=str(np.around(mass, 3))):
                mom = self.momenta[idx]
                proj_verts = self.projected_vertices(
                    mass, mom, momvar_idx, subscheme)
                for current, vertex in proj_verts.items():
                    if current in Zs:
                        Zs[current].append(vertex**(-1))
                    else:
                        Zs[current] = [vertex**(-1)]

            Zs = {current: join_stats(Z) for current, Z in Zs.items()}

            self.save_Z_bls(Zs, mass, momvar_idx, subscheme)

        else:
            Zs = self.load_Z_bls(mass, momvar_idx, subscheme)

        if plot:
            sublabel = r'\gamma_\mu' if subscheme == 'gamma' else r'\not{q}'
            title = self.scheme+r'$^{'+sublabel+r'}$, $m_\pi=' +\
                str(np.around(mass, 3))+r'$ mom combo '+str(momvar_idx+1)
            fname = f'plots/{self.ens.name}_Zs_{self.mass_map[mass]}_tw{momvar_idx}.pdf'
            self.plot_Z_bls(Zs, title, fname)

        return Zs

    def load_Z_bls(self, mass: float, momvar_idx: int,
                   subscheme: str) -> Dict:

        file = h5py.File(self.Zdata_fname, 'a')
        grp_name = f'{subscheme}/Bilinear/{self.mass_map[mass]}' +\
            f'/momvar_{momvar_idx+1}'

        grp = file[grp_name]
        self.momenta = list(np.array(grp['ap'][:])*self.ens.ainv)
        Zs = {}
        for current in self.currents:
            Zs[current] = Stat(
                val=grp[f'{current}/central'][:],
                err=grp[f'{current}/errors'][:],
                btsp=grp[f'{current}/bootstrap'][:]
            )
        return Zs

    def save_Z_bls(self, Zs: Dict, mass: float,
                   momvar_idx: int, subscheme: str) -> None:

        file = h5py.File(self.Zdata_fname, 'a')
        grp_name = f'{subscheme}/Bilinear/{self.mass_map[mass]}' +\
            f'/momvar_{momvar_idx+1}'

        if grp_name in file.keys():
            del file[grp_name]

        grp = file.create_group(grp_name)
        grp.attrs['momentum_variation'] = mom_combos[momvar_idx]

        grp.create_dataset('ap', data=np.array(self.momenta)/self.ens.ainv)
        for current in self.currents:
            grp.create_dataset(f'{current}/central', data=Zs[current].val)
            grp.create_dataset(f'{current}/errors', data=Zs[current].err)
            grp.create_dataset(f'{current}/bootstrap', data=Zs[current].btsp)

        print(f'saved data to {grp_name} in {self.Zdata_fname}')

    def plot_Z_bls(self, Zs: Dict, title: str, fname: str) -> None:

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(3, 5),
                               gridspec_kw={"height_ratios": [2, 1]})
        plt.subplots_adjust(hspace=0)

        for idx, current in enumerate(self.currents):
            ax[0].errorbar(self.momenta, Zs[current].val,
                           yerr=Zs[current].err, fmt='o',
                           capsize=4, label=current)
        ax[0].set_ylabel(r'$Z_\Gamma/Z_q$')
        ax[0].legend()

        for X, Y in zip(['S', 'V'], ['P', 'A']):
            ys = Zs[X]/Zs[Y]
            ax[1].errorbar(self.momenta, ys.val, ys.err,
                           fmt='o', capsize=4, label=f'{X}/{Y}')
        ax[1].set_ylabel(r'$\chi$SB')
        ax[1].axhline(1.0, c='k', linestyle='dashed')
        ax[1].set_xlabel(r'$\sqrt{q^2}$ [GeV]')
        ax[1].set_ylim([0.5, 1.5])
        ax[1].legend()

        ax[0].set_title(title)
        callPDF(fname, show=False)
        print(f'plotted to {os.getcwd()}/{fname}')

    def projected_vertices(self, mass: float, mom: float,
                           momvar_idx: int, subscheme: str) -> Dict:

        mass_str, mom_str = self.mass_map[mass], self.mom_map[mom]
        theta_in, theta_out = self.theta_str[mass_str][mom_str][momvar_idx]

        operators = self.load_bl_operators(
            mass, mom, theta_in, theta_out)

        if subscheme not in ['gamma', 'qslash']:
            raise 'Need to pass in subscheme gamma or qslash (str)'

        mom_in = convert_to_phys(theta_in, self.ens.L, self.ens.T)
        mom_out = convert_to_phys(theta_out, self.ens.L, self.ens.T)
        qvec = mom_in - mom_out

        projectors = bilinear_projectors(subscheme, qvec=qvec)

        return {current: join_stats([Stattensordot(
            projectors[current][idx], operators[current][idx]).use_func(tensortrace).use_func(np.real)
            for idx in range(len(projectors[current]))]).use_func(np.sum, axis=0)
            for current in projectors.keys()
        }

    def load_bl_operators(self, mass: float, mom: float,
                          theta_in: np.ndarray, theta_out: np.ndarray) -> Dict:
        amputees = self.load_amputated_bls(
            mass, mom, theta_in, theta_out)

        return {
            "S": [amputees[TwoPointFn.vertices.index('Identity')]],
            "P": [amputees[TwoPointFn.vertices.index('Gamma5')]],
            "V": [amputees[TwoPointFn.vertices.index(f'Gamma{mu}')] for mu in dirs],
            "A": [amputees[TwoPointFn.vertices.index(f'Gamma{mu}Gamma5')] for mu in dirs],
            "T": sum([[amputees[TwoPointFn.vertices.index(f'Sigma{dirs[mu]}{dirs[nu]}')]
                       for nu in range(mu+1, N_dir)]
                      for mu in range(N_dir-1)], [])
        }

    def load_amputated_bls(self, mass: float, mom: float,
                           theta_in: np.ndarray, theta_out: np.ndarray) -> np.ndarray:

        bilinears = self.load_bls(mass, mom, theta_in, theta_out)

        mass_str, mom_str = self.mass_map[mass], self.mom_map[mom]
        in_prop = load_external_leg(self.ens.name, mass_str, mom_str, theta_in)
        out_prop = load_external_leg(
            self.ens.name, mass_str, mom_str, theta_out)

        return np.array([bl_leg_ampute(bilinears[b], in_prop, out_prop)
                         for b in range(len(TwoPointFn.vertices))])

    def load_bls(self, mass: float, mom: float,
                 theta_in: np.ndarray, theta_out: np.ndarray) -> np.ndarray:
        # reads in data over all configs for a given momentum combination

        theta_in_str = '_'.join(theta_in)
        theta_out_str = '_'.join(theta_out)

        files = [f'{self.path}/{self.mass_map[mass]}/{self.mom_map[mom]}' +
                 f'/{self.prefix}{theta_in_str}_{theta_out_str}.{cf}.h5'
                 for cf in self.cf_list]

        data = np.empty(shape=(self.N_cf, len(TwoPointFn.vertices), N_dir,
                               N_dir, N_col, N_col), dtype='complex128')

        for cf in range(self.N_cf):
            try:
                file = h5py.File(files[cf], 'r')['Bilinear']
            except OSError:
                print(files[cf])
                pdb.set_trace()
            for vx in range(len(TwoPointFn.vertices)):
                corr = file[f'Bilinear_{vx}']['corr'][0, 0, :]
                data[cf, vx] = np.array(corr["re"]+1j*corr["im"])

        bilinears = np.array([Stat(
            val=np.mean(data[:, b_idx], axis=0),
            err='fill',
            btsp=bootstrap(data[:, b_idx], seed=self.ens.name)
        ) for b_idx in range(len(TwoPointFn.vertices))], dtype=object)

        return bilinears

    def get_bl_list(self, path: str) -> np.ndarray:
        # get the list of bilinear momentum combinations

        all_files = [f for f in os.listdir(path) if f.startswith(self.prefix)]
        mom_combinations = []
        for f in all_files:
            config, mom1, mom2 = decode_bl_fname(f)
            if [mom1, mom2] in mom_combinations:
                continue
            else:
                partial_str = f'{self.prefix}' +\
                    '_'.join(mom1)+'_'+'_'.join(mom2)
                other_configs = [
                    f for f in all_files if f.startswith(partial_str)]
                if len(other_configs) == self.N_cf:
                    mom_combinations.append([mom1, mom2])
                else:
                    print(f'only {len(other_configs)} config files' +
                          f' found for ({mom1}, {mom2}) in {path}\n')

        return mom_combinations

    def create_attributes(self) -> None:
        self.theta_str = {mass_str: {mom_str: self.get_bl_list(f'{self.path}/{mass_str}/{mom_str}')
                                     for mom_str in os.listdir(f'{self.path}/{mass_str}')}
                          for mass_str in os.listdir(self.path)}

        self.mass_map = {mass_str2float(
            mass): mass for mass in self.theta_str.keys()}
        self.masses = sorted(list(self.mass_map.keys()))

        self.mom_map = {np.linalg.norm(convert_to_phys(theta[0][0],
                        self.ens.L, self.ens.T))*self.ens.ainv: mom_str
                        for mom_str, theta in self.theta_str[self.mass_map[self.masses[0]]].items()}
        self.momenta = sorted(list(self.mom_map.keys()))


def chiral_ansatz(mpis, param, **kwargs):
    return param[0] + param[1]*mpis**2


def convert_to_phys(vec: np.ndarray, L: int, T: int) -> np.ndarray:
    vec = np.array(list(map(float, vec)))
    L, T = L/(2*np.pi), T/(2*np.pi)
    return np.array(list(vec[:3]/L)+[vec[-1]/T])


def load_external_leg(ensemble: str, mass_str: str, mom_str: str,
                      theta: np.ndarray) -> Stat:
    """ Given theta, reads in data for external leg"""

    ens = Ensemble(ensemble)
    prefix = 'ExternalLeg_0_'
    path = ens.datafolder+'/npr_data'
    theta_str = '_'.join(theta)
    cf_list = pars[ensemble]['NPR_cfgs']
    N_cf = len(cf_list)

    data = np.empty(
        shape=(N_cf, N_dir, N_dir, N_col, N_col), dtype='complex128')

    for cf in range(N_cf):
        fname = f'{path}/{mass_str}/{mom_str}/{prefix}{theta_str}.{cf_list[cf]}.h5'
        try:
            corr = h5py.File(fname, 'r')['ExternalLeg']['corr'][0, 0, :]
        except OSError:
            print(fname)
            pdb.set_trace()
        data[cf] = np.array(corr["re"]+1j*corr["im"])

    return Stat(
        val=np.mean(data, axis=0),
        err='fill',
        btsp=bootstrap(data, seed=ens.name)
    )


def bl_leg_ampute(bilinear: Stat, in_prop: Stat, out_prop: Stat) -> Stat:
    in_prop_inv = in_prop.use_func(tensorinv)
    out_prop_G5_conj = out_prop.use_func(G5H)
    out_prop_inv = out_prop_G5_conj.use_func(tensorinv)

    return Stattensordot(out_prop_inv, Stattensordot(bilinear, in_prop_inv))


def decode_bl_fname(fname: str) -> Tuple[int, List, List]:
    components = fname.rsplit('_')
    mom1 = components[3:7]
    mom2 = components[7:10]
    cfg = components[-1].rsplit('.')[-2]
    num = components[-1].rsplit('.'+cfg)[0]

    mom2.append(num)
    return int(cfg), mom1, mom2


def mom_combo_sort(arr: np.ndarray) -> np.ndarray:
    """ sorts momentum combinations in the form
        [[A,A,0,0], [B,0,B,0],
         [A,A,A,A], [0,0,0,B],
         [A,A,A,A], [B,B,B,B]]
    """
    new_arr = np.empty(shape=arr.shape)
    for i in range(3):
        A, B = arr[i, 0], arr[i, 1]
        if np.all(A != 0):
            idx = 1 if np.count_nonzero(B) == 1 else 2
        else:
            idx = 0
        new_arr[idx, 0], new_arr[idx, 1] = A, B

    return new_arr


mom_combos = [
    r'A_A_0_0__B_0_B_0',
    r'A_A_A_A__0_0_0_B',
    r'A_A_A_A__B_B_B_B'
]
