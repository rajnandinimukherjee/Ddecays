from projectors import *
from pion_mass import *


class Bilinear:
    currents = ['S', 'P', 'V', 'A', 'T']

    def __init__(self, ensemble: str, scheme: str = 'SMOM',
                 compute: bool = False) -> None:
        self.ens = Ensemble(ensemble)
        self.scheme = scheme
        self.prefix = f'{scheme}_Bilinear_00_'
        self.path = f'{self.ens.path}/new_runs/{self.ens.dataname}/npr_data'
        self.compute = compute

        self.Zdata_fname = f'Z_factors/{self.ens.dataname}.hd5'

        if self.compute:
            self.mass_map, self.cf_list = self.ens.config_counter(
                data='NPR', prefix=f'{scheme}_Bilinear', show=False)
            self.N_cf = len(self.cf_list)
            self.masses = sorted(list(self.mass_map.keys()))
            self.create_attributes()
        else:
            self.mass_map = {mass_str2float(mass_str): mass_str
                             for mass_str in h5py.File(self.Zdata_fname, 'r')
                             ['Bilinear'].keys() if mass_str != 'm0p0'}
            self.masses = sorted(list(self.mass_map.keys()))

    def plot_chiral_extrap_allmom(self, subscheme: str) -> None:
        fig, ax = plt.subplots(nrows=len(self.currents),
                               ncols=1, figsize=(3, 10))
        plt.subplots_adjust(hspace=0)
        sublabel = r'\gamma_\mu' if subscheme == 'gamma' else r'\not{q}'
        title = self.scheme+r'$^{'+sublabel + \
            r'}$, $m_\pi=0$, all combos'
        ax[0].set_title(title)

        for momvar_idx in range(3):
            Zs = self.load_chiral_extrap(momvar_idx, subscheme, plot=False)
            for c_idx, current in enumerate(self.currents):
                ax[c_idx].errorbar(self.momenta, Zs[current].val,
                                   yerr=Zs[current].err, fmt='o',
                                   capsize=4, label=mom_combos[momvar_idx])
                if momvar_idx == 0:
                    ax[c_idx].set_ylabel(r'$Z_'+current+r'/Z_q$')

        ax[-1].set_xlabel(r'$\sqrt{q^2}$ [GeV]')
        handles, labels = ax[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

        fname = f'plots/{self.ens.name}_Zs_chiral_extrap_all_tw.pdf'
        callPDF(fname, show=False)
        print(f'plotted to {os.getcwd()}/{fname}')

    def load_chiral_extrap(self, momvar_idx: int, subscheme: str,
                           plot: bool = False) -> Dict:

        self.compute = False
        Zs = self.get_all_Zs(momvar_idx, subscheme, plot=False)
        self.pion = TwoPointFn(self.ens.name, compute=False)
        self.pion_masses = join_stats(self.pion.load_meson_masses())

        file = h5py.File(self.Zdata_fname, 'r')
        grp_name = f'Bilinear/m0p0/{subscheme}' +\
            f'/momvar_{momvar_idx+1}'
        grp = file[grp_name]

        extrap = {current: Stat(
            val=np.array(grp[f'{current}/central'][:]),
            err=np.array(grp[f'{current}/errors'][:]),
            btsp=np.array(grp[f'{current}/bootstrap'][:])
        ) for current in self.currents}
        file.close()

        if plot:
            fig, ax = plt.subplots(nrows=len(self.currents),
                                   ncols=1, figsize=(3, 10))
            plt.subplots_adjust(hspace=0)
            sublabel = r'\gamma_\mu' if subscheme == 'gamma' else r'\not{q}'
            title = self.scheme+r'$^{'+sublabel + \
                r'}$, mom combo '+str(momvar_idx+1)
            ax[0].set_title(title)

            for c_idx, current in enumerate(self.currents):
                for m_idx, mass in enumerate(self.masses):
                    pion_label = err_disp(self.pion_masses.val[m_idx],
                                          self.pion_masses.err[m_idx])
                    ax[c_idx].errorbar(self.momenta, Zs[mass][current].val,
                                       yerr=Zs[mass][current].err, fmt='o',
                                       capsize=4, label='$m_\pi='+pion_label+'$')
                ax[c_idx].set_ylabel(r'$Z_'+current+r'/Z_q$')
                ax[c_idx].errorbar(self.momenta, extrap[current].val,
                                   yerr=extrap[current].err, fmt='o',
                                   capsize=4, label=f'extrap', c='k')

            ax[-1].set_xlabel(r'$\sqrt{q^2}$ [GeV]')
            handles, labels = ax[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right')
            title = self.scheme+r'$^{'+sublabel + \
                r'}$, $m_\pi=0$ mom combo '+str(momvar_idx+1)
            fname = f'plots/{self.ens.name}_Zs_chiral_extrap_tw{momvar_idx}.pdf'
            callPDF(fname, show=False)
            print(f'plotted to {os.getcwd()}/{fname}')

        return extrap

    def quick_chiral_plot(self, mpis: Stat, Zs: Stat,
                          res: Stat, title: str) -> None:

        fig, ax = plt.subplots()
        x = mpis**2
        ax.errorbar(x.val, Zs.val, xerr=x.err, yerr=Zs.err,
                    fmt='o', capsize=4, label=r'$Z(am_q)$')
        ax.errorbar([0.0], [res.val[0]], yerr=[res.err[0]],
                    fmt='o', capsize=4, label=r'$Z(am_q=0)$')
        ax.axvline(0, color='k', ls='dashed')
        xmin, xmax = ax.get_xlim()
        xrange = np.linspace(-0.05, mpis.val[-1], 50)
        yrange = res.mapping(xrange)
        ax.fill_between(xrange**2, yrange.val+yrange.err,
                        yrange.val-yrange.err, color='k',
                        alpha=0.2, label=r'fit')
        ax.text(0.5, 0.1, r'$\chi^2/$DOF$=' +
                str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                va='center', ha='center', transform=ax.transAxes)
        ax.set_xlim([xmin, xmax])
        ax.set_xlabel(r'$m_\pi^2$ [GeV${}^2$]')
        ax.set_ylabel(r'$Z_\Gamma/Z_q$')
        ax.set_title(title)
        ax.legend()

    def chiral_extrap(self, momvar_idx: int, subscheme: str,
                      plot: bool = False, save: bool = True) -> Dict:

        self.compute = False
        Zs = self.get_all_Zs(momvar_idx, subscheme, plot=False)
        self.pion = TwoPointFn(self.ens.name, compute=False)

        if np.all(self.masses == self.pion.masses):
            self.pion_masses = join_stats(self.pion.load_meson_masses())
        else:
            print(
                f"mismatch between NPR masses {self.masses} and \nvalence masses {self.pion.masses}")
            return None

        extrap = {}

        for c_idx, current in enumerate(self.currents):
            extrap[current] = []
            for m_idx in tqdm(range(len(self.momenta)),
                              leave=False, desc=current):
                mom = self.momenta[m_idx]
                ys = join_stats([Zs[mass][current][m_idx]
                                 for mass in self.masses])
                res = fit_func(self.pion_masses, ys, chiral_ansatz,
                               [1, 1], correlated=False)
                if current in ['V', 'A'] and m_idx < 5 and plot:
                    self.quick_chiral_plot(
                        self.pion_masses,
                        ys, res,
                        r'chiral extrap $\mu='+str(np.around(mom, 3)) +
                        r'$, $\Gamma=$'+current
                    )
                extrap[current].append(res[0])
            extrap[current] = join_stats(extrap[current])

        if save:
            file = h5py.File(self.Zdata_fname, 'a')
            grp_name = f'Bilinear/m0p0/{subscheme}' +\
                f'/momvar_{momvar_idx+1}'

            if grp_name in file.keys():
                del file[grp_name]

            grp = file.create_group(grp_name)
            grp.attrs['momentum_variation'] = mom_combos[momvar_idx]

            grp.create_dataset('ap', data=np.array(self.momenta)/self.ens.ainv)
            for current in self.currents:
                grp.create_dataset(f'{current}/central',
                                   data=extrap[current].val)
                grp.create_dataset(f'{current}/errors',
                                   data=extrap[current].err)
                grp.create_dataset(f'{current}/bootstrap',
                                   data=extrap[current].btsp)

            print(f'saved data to {grp_name} in {self.Zdata_fname}')

        if plot:
            sublabel = r'\gamma_\mu' if subscheme == 'gamma' else r'\not{q}'
            title = self.scheme+r'$^{'+sublabel + \
                r'}$, $m_\pi=0$ mom combo '+str(momvar_idx+1)
            fname = f'plots/{self.ens.name}_Zs_chiral_extrap_tw{momvar_idx}.pdf'
            self.plot_Z_bls(extrap, title, fname)
        return extrap

    def get_all_Zs(self, momvar_idx: int, subscheme: str,
                   plot: bool = True) -> Dict:

        Zs = {mass: self.get_Zs(mass, momvar_idx, subscheme)
              for mass in self.masses}

        if plot:
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
            title = self.scheme+r'$^{'+sublabel+r'}$, $am_q=' +\
                str(np.around(mass, 3))+r'$ mom combo '+str(momvar_idx+1)
            fname = f'plots/{self.ens.name}_Zs_{self.mass_map[mass]}_tw{momvar_idx}.pdf'
            self.plot_Z_bls(Zs, title, fname)

        return Zs

    def load_Z_bls(self, mass: float, momvar_idx: int,
                   subscheme: str) -> Dict:

        file = h5py.File(self.Zdata_fname, 'a')
        grp_name = f'Bilinear/{self.mass_map[mass]}/{subscheme}' +\
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
        grp_name = f'Bilinear/{self.mass_map[mass]}/{subscheme}' +\
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
            raise 'Subscheme is either gamma or qslash'

        mom_in = convert_to_phys(theta_in, self.ens.L, self.ens.T)
        mom_out = convert_to_phys(theta_out, self.ens.L, self.ens.T)
        qvec = mom_in - mom_out

        projectors = bilinear_projectors(subscheme, qvec=qvec)

        return {current: sum([projectors[current][i]@operators[current][i]
                              for i in range(len(operators[current]))], Zero).
                use_func(np.trace).use_func(np.real)
                for current in projectors.keys()}

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
        in_prop = self.load_external_leg(mass, mom, theta_in)
        out_prop = self.load_external_leg(mass, mom, theta_out)
        out_prop_g5 = out_prop.use_func(g5)

        in_prop_inv = in_prop.use_func(np.linalg.inv)
        out_prop_inv = out_prop_g5.use_func(np.linalg.inv)

        return np.array([out_prop_inv@bilinears[b]@in_prop_inv
                         for b in range(len(TwoPointFn.vertices))])

    def load_bls(self, mass: float, mom: float,
                 theta_in: np.ndarray, theta_out: np.ndarray) -> np.ndarray:
        # reads in data over all configs for a given momentum combination

        theta_in_str = '_'.join(theta_in)
        theta_out_str = '_'.join(theta_out)

        files = [f'{self.path}/{self.mass_map[mass]}/{self.mom_map[mom]}' +
                 f'/{self.prefix}{theta_in_str}_{theta_out_str}.{cf}.h5'
                 for cf in self.cf_list]

        data = np.empty(shape=(self.N_cf, len(TwoPointFn.vertices),
                               N_cd, N_cd), dtype='complex128')

        for cf in range(self.N_cf):
            try:
                file = h5py.File(files[cf], 'r')['Bilinear']
            except OSError:
                print(files[cf])
                pdb.set_trace()
            for vx in range(len(TwoPointFn.vertices)):
                corr = file[f'Bilinear_{vx}']['corr'][0, 0, :]

                data[cf, vx] = np.array(
                    corr["re"]+1j*corr["im"]).swapaxes(1, 2).reshape(
                    (N_cd, N_cd), order='F')

        bilinears = np.array([Stat(
            val=np.mean(data[:, b_idx], axis=0),
            btsp=bootstrap(data[:, b_idx], seed=self.ens.seed)
        ) for b_idx in range(len(TwoPointFn.vertices))], dtype=object)

        return bilinears

    def load_external_leg(self, mass: float, mom: float,
                          theta: np.ndarray) -> Stat:
        """ Given theta, reads in data for external leg"""

        prefix = 'ExternalLeg_0_'
        theta_str = '_'.join(theta)

        files = [f'{self.path}/{self.mass_map[mass]}/{self.mom_map[mom]}' +
                 f'/{prefix}{theta_str}.{cf}.h5' for cf in self.cf_list]

        data = np.empty(
            shape=(self.N_cf, N_cd, N_cd), dtype='complex128')

        for cf in range(self.N_cf):
            try:
                corr = h5py.File(files[cf], 'r')[
                    'ExternalLeg']['corr'][0, 0, :]
            except OSError:
                print(fname)
                pdb.set_trace()

            data[cf] = np.array(corr["re"]+1j*corr["im"]).swapaxes(1, 2).reshape(
                (N_cd, N_cd), order='F')

        externalleg = Stat(
            val=np.mean(data, axis=0),
            btsp=bootstrap(data, seed=self.ens.seed)
        )

        return externalleg

    def get_bl_list(self, path: str) -> np.ndarray:
        # get the list of bilinear momentum combinations

        all_files = [f for f in os.listdir(path) if f.startswith(self.prefix)]
        mom_combinations = []
        for f in all_files:
            config, mom1, mom2 = decode_bl_fname(f)
            if [mom1, mom2] in mom_combinations:
                continue
            else:
                partial_str = f'{self.prefix}' + \
                    '_'.join(mom1)+'_'+'_'.join(mom2)
                other_configs = [
                    f for f in all_files if f.startswith(partial_str)]
                if len(other_configs) == self.N_cf:
                    mom_combinations.append([mom1, mom2])
                else:
                    print(f'only {len(other_configs)} config files' +
                          f' found for ({mom1}, {mom2}) in {path}\n')

        return mom_combo_sort(np.array(mom_combinations))

    def create_attributes(self) -> None:
        self.theta_str = {mass_str: {mom_str: self.get_bl_list(f'{self.path}/{mass_str}/{mom_str}')
                                     for mom_str in os.listdir(f'{self.path}/{mass_str}')}
                          for mass, mass_str in self.mass_map.items()}

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


mom_combos = [
    r'A_A_0_0__B_0_B_0',
    r'A_A_A_A__0_0_0_B',
    r'A_A_A_A__B_B_B_B'
]
