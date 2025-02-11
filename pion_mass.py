from ensembles import *


class TwoPointFn:
    vertices = ['Identity', 'Gamma5',
                'GammaX', 'GammaY', 'GammaZ', 'GammaT',
                'GammaXGamma5', 'GammaYGamma5', 'GammaZGamma5', 'GammaTGamma5',
                'SigmaXT', 'SigmaXY', 'SigmaXZ', 'SigmaYT', 'SigmaYZ', 'SigmaZT']

    def __init__(self, ensemble: str, gamma_src: str = 'Gamma5',
                 gamma_snk: str = 'Gamma5', compute: bool = True) -> None:
        self.ens = Ensemble(ensemble)
        if gamma_src in self.vertices and gamma_snk in self.vertices:
            self.gamma_src, self.gamma_snk = gamma_src, gamma_snk
        else:
            raise "Need to specify src and snk gamma structure. " +\
                "See TwoPointFn.vertices for accepted str formats."

        self.path = self.ens.path + \
            f'/hadronic_ward_identity/{self.ens.dataname}/s0g0'

        self.Zdata_fname = f'Z_factors/{self.ens.dataname}.hd5'

        self.compute = compute
        if self.compute:
            self.mass_map, self.cf_list = self.ens.config_counter(
                data='valence', prefix=f'two_point', show=False)
            self.N_cf = len(self.cf_list)
        else:
            self.mass_map = {mass_str2float(mass_str): mass_str
                             for mass_str in h5py.File(self.Zdata_fname, 'r')['Pion'].keys()}
        self.masses = sorted(list(self.mass_map.keys()))

    def load_meson_masses(self, plot: bool = False, **kwargs) -> List:

        mesons = []
        for mass in self.masses:

            file = h5py.File(self.Zdata_fname, 'r')
            grp_name = f'Pion/{self.mass_map[mass]}/fit'
            mpi = Stat(
                val=np.array(file[grp_name+'/central']),
                err=np.array(file[grp_name+'/errors']),
                btsp=np.array(file[grp_name+'/bootstrap']),
            )
            mpi.chi_sq = file[grp_name].attrs['chi_sq']
            mpi.DOF = file[grp_name].attrs['DOF']
            mesons.append(mpi)

        if plot:
            x = np.array(self.masses)
            y = join_stats(mesons)**2
            if 'ax' not in kwargs:
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
                plt.subplots_adjust(hspace=0)

                ax[0].errorbar(x, y.val, yerr=y.err,
                               capsize=4, fmt='o')
                ax[0].set_ylabel(r'$m_\pi^2$ [GeV${}^2$]')

                chis = [mpi.chi_sq/mpi.DOF for mpi in mesons]
                ax[1].scatter(x, chis, marker='x')
                ax[1].set_ylabel(r'$\chi^2/\mathrm{DOF}$')
                ax[1].set_xlabel(r'$am_q$')
                fname = f'plots/{self.ens.name}_mpi_variation.pdf'
                callPDF(fname, show=False)
                print(f'plot saved to {fname}')
            else:
                ax.errorbar(x, y.val, yerr=y.err,
                            capsize=4, fmt='o')

        return mesons

    def fit_meson(self, mass: float, fitrange: np.ndarray,
                  ansatz: str = 'cosh', plot: bool = True,
                  plotrange=None, save: bool = True) -> Stat:

        if self.compute:
            data, twopf = self.load_meson(mass, save=save)
        else:
            twopf = self.read_meson(mass)

        meff = twopf.use_func(m_eff, ansatz=ansatz)*self.ens.ainv

        res = fit_func(fitrange, meff[fitrange],
                       constant_ansatz, [1, 0], correlated=False)
        mpi = res[0]
        mpi.chi_sq, mpi.DOF, mpi.pvalue = res.chi_sq, res.DOF, res.pvalue

        if plot:
            fig, ax = plt.subplots()
            if type(plotrange) is not np.ndarray:
                plotrange = np.arange(1, int(self.ens.T/2))
                if ansatz == 'cosh':
                    plotrange = plotrange[:-1]
                y = meff
            else:
                y = meff[plotrange]

            ax.errorbar(plotrange, y.val, yerr=y.err,
                        capsize=4, fmt='o', label='data')
            ax.fill_between(fitrange, mpi.val-mpi.err,
                            mpi.val+mpi.err, color='k',
                            alpha=0.1, label=r'$m_\pi$ fit')
            ax.set_xlabel(r'$t/a$')
            ax.set_ylabel(r'$m_\mathrm{eff}^\mathrm{'+ansatz+r'}$ [GeV]')
            ax.set_title(r'$am_q=-'+str(np.around(mass, 3))+r'$')
            ax.legend()
            ax.text(0.5, 0.1, r'$\chi^2/$DOF$=' +
                    str(np.around(res.chi_sq/res.DOF, 3))+r'$',
                    va='center', ha='center', transform=ax.transAxes)

            fname = f'plots/{self.ens.name}_mpi_fit_{self.mass_map[mass]}.pdf'
            callPDF(fname, show=False)
            print(f'plot saved to {fname}')

        if save:
            file = h5py.File(self.Zdata_fname, 'a')
            grp_name = f'Pion/{self.mass_map[mass]}/fit'

            if grp_name in file.keys():
                del file[grp_name]

            grp = file.create_group(grp_name)
            grp.create_dataset('central', data=mpi.val)
            grp.create_dataset('errors', data=mpi.err)
            grp.create_dataset('bootstrap', data=mpi.btsp)

            grp.attrs['fitrange'] = fitrange
            grp.attrs['chi_sq'] = res.chi_sq
            grp.attrs['DOF'] = res.DOF

            print(f'saved fit_results to {grp_name} in {self.Zdata_fname}')

    def read_meson(self, mass: float) -> Stat:

        file = h5py.File(self.Zdata_fname, 'a')
        grp_name = f'Pion/{self.mass_map[mass]}/corr'

        twopf = Stat(
            val=np.array(file[grp_name]['central'][:]),
            err=np.array(file[grp_name]['errors'][:]),
            btsp=np.array(file[grp_name]['bootstrap'][:])
        )
        return twopf

    def plot_cfgs(self, mass: float, plotrange: np.ndarray) -> None:

        data, corr = self.load_meson(mass)
        fig, ax = plt.subplots()
        for c_idx, cf in enumerate(self.cf_list):
            ax.plot(plotrange[1:-1],
                    m_eff(data[c_idx, plotrange]), label=str(cf))
        ymin, ymax = ax.get_ylim()

        meff_corr = corr.use_func(m_eff)
        ax.errorbar(np.arange(2, len(corr.val)), meff_corr.val,
                    yerr=meff_corr.err, c='k', capsize=4, fmt='o',
                    label='btsp')
        ax.legend()
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([plotrange[0], plotrange[-1]])
        ax.set_ylabel(r'$m_\mathrm{eff}^\mathrm{2pt}(t,\mathrm{cf})$')
        ax.set_xlabel(r'$t/a$')
        ax.set_title(r'$am_q=-'+str(np.around(mass, 3))+r'$')

        fname = f'plots/{self.ens.name}_pion2pf_cfgs.pdf'
        callPDF(fname, show=False)
        print(f'plot saved to {fname}')

    def load_meson(self, mass: float, save: bool = True) -> Tuple[List, Stat]:

        files = [f'{self.path}/{self.mass_map[mass]}/mesons/two_point_0.{cf}.h5'
                 for cf in self.cf_list]

        try:
            meson_grp = find_meson_group(
                files[0], self.gamma_src, self.gamma_snk)
        except OSError:
            pdb.set_trace()

        data = np.zeros(shape=(self.N_cf, self.ens.T))

        for cf in range(self.N_cf):
            try:
                file = h5py.File(files[cf], 'r')['meson'][meson_grp]
            except OSError:
                print(files[cf])
                pdb.set_trace()
            for vx in range(len(TwoPointFn.vertices)):
                corr = file['corr'][:]
                data[cf, :] = np.array(corr["re"])

        twopf = Stat(
            val=np.mean(data, axis=0),
            err='fill',
            btsp=bootstrap(data, seed=self.ens.name),
        )

        # fold
        twopf = (twopf[1:]+twopf[::-1][:-1])[:int(self.ens.T/2)]*0.5

        if save:
            file = h5py.File(self.Zdata_fname, 'a')
            grp_name = f'Pion/{self.mass_map[mass]}/corr'

            if grp_name in file.keys():
                del file[grp_name]

            grp = file.create_group(grp_name)
            grp.create_dataset('central', data=twopf.val)
            grp.create_dataset('errors', data=twopf.err)
            grp.create_dataset('bootstrap', data=twopf.btsp)

            print(f'saved data to {grp_name} in {self.Zdata_fname}')

        return data, twopf


def constant_ansatz(t, param, **kwargs):
    return param[0]*np.ones(len(t))


def linear_ansatz(t, param, **kwargs):
    return param[0] + param[1]*t


def find_meson_group(fname: str, gamma_src: str, gamma_snk: str) -> str:
    file = h5py.File(fname, 'r')['meson']
    for group in file.keys():
        if file[group].attrs['gamma_src'][0].decode('utf-8') == gamma_src:
            if file[group].attrs['gamma_snk'][0].decode('utf-8') == gamma_snk:
                return group
    raise f'{gamma_src}x{gamma_snk} meson group not found in {fname}.'


def closest_n_points(self, target: float, values: np.ndarray, n: int) -> List:
    diff = list(np.abs(values-target))
    sort = sorted(diff)
    closest_idx = []
    for n_idx in range(n):
        nth_closest_point = diff.index(sort[n_idx])
        closest_idx.append[nth_closest_point]
    return closest_idx
