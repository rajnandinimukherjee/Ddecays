from bilinears import *


class Fourquark:
    vertices = [
        [f"Gamma{a}", f"Gamma{b}"]
        for x in dirs
        for a in (x, x + "Gamma5")
        for b in (x, x + "Gamma5")
    ]
    decay_vertices = ["VApAVS+", "VApAVS-"]
    decay_vtx_str = [r"_{[VA+AV]}^{\mathcal{S}=+}",
                     r"_{[VA+AV]}^{\mathcal{S}=-}"]

    def __init__(
        self,
        ensemble: str,
        scheme: str = "SMOM",
        compute: bool = False,
        norm: str = "A",
    ) -> None:
        self.ens = Ensemble(ensemble)
        self.scheme = scheme
        self.prefix = f"{scheme}_FourQuark_00_"
        self.path = f"{self.ens.path}/new_runs/{self.ens.dataname}/npr_data"
        self.compute = compute
        self.N_tw = 3 if scheme == "SMOM" else 5
        self.mom_combos = SMOM_combos if scheme == "SMOM" else MOM_combos
        self.norm = norm

        self.Zdata_fname = f"Z_factors/{self.ens.dataname}_{self.scheme}.hd5"

        if self.compute:
            self.mass_map, self.cf_list = self.ens.config_counter(
                data="NPR", prefix=f"{scheme}_FourQuark", show=False
            )
            self.N_cf = len(self.cf_list)
            self.masses = sorted(list(self.mass_map.keys()))
            self.create_attributes()
        else:
            self.mass_map = {
                mass_str2float(mass_str): mass_str
                for mass_str in h5py.File(self.Zdata_fname, "r")["Fourquark"].keys()
                if mass_str != "m0p0"
            }
            self.masses = sorted(list(self.mass_map.keys()))
            self.aqvecs = {}

        self.bilinear = Bilinear(ensemble, compute=False, scheme=self.scheme)
        self.twist_diffs = {}

    def plot_twist_diffs(
        self,
        sub_idx: int = 0,
        subscheme: str = "gamma",
        show: bool = False,
        subtract_q6: bool = True,
        num_params: int = 2,
        zoom: bool = False,
        **kwargs,
    ) -> None:

        self.twist_diffs[sub_idx + 1] = {}

        fig, ax = plt.subplots(nrows=len(self.decay_vertices), sharex=True)
        plt.subplots_adjust(hspace=0)
        sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
        title = self.scheme + r"$^{" + sublabel + \
            r"}$, $m_\pi=0$, twist differences"
        plt.suptitle(title)

        Zs_sub = self.load_chiral_extrap(sub_idx, subscheme, plot=False)
        aq4_sub = np.sum(self.aqvecs[sub_idx + 1] ** 4, axis=1)
        aq6_sub = np.sum(self.aqvecs[sub_idx + 1] ** 6, axis=1)
        for momvar_idx in range(self.N_tw):
            if momvar_idx != sub_idx:
                self.twist_diffs[sub_idx + 1][momvar_idx + 1] = {}
                Zs = self.load_chiral_extrap(momvar_idx, subscheme, plot=False)
                aq4 = np.sum(self.aqvecs[momvar_idx + 1] ** 4, axis=1)
                aq6 = np.sum(self.aqvecs[momvar_idx + 1] ** 6, axis=1)
                del_aq4 = aq4 - aq4_sub
                del_aq6 = aq6 - aq6_sub

                sign = del_aq4[0] / np.abs(del_aq4[0])
                for v_idx, vertex in enumerate(self.decay_vertices):
                    diff = Zs[vertex] - Zs_sub[vertex]
                    label = (
                        f"twist ${momvar_idx+1}-{sub_idx+1}$"
                        if sign > 0
                        else f"twist ${sub_idx+1}-{momvar_idx+1}$"
                    )
                    ax[v_idx].errorbar(
                        np.abs(del_aq4),
                        # self.momenta,
                        diff.val,
                        yerr=diff.err,
                        fmt="o",
                        capsize=4,
                        color=pltcolors[momvar_idx],
                        alpha=0.3 if subtract_q6 else 1.0,
                        label=None if subtract_q6 else label,
                    )
                    ylabel = r"$Z" + self.decay_vtx_str[v_idx]
                    if self.norm!="q"
                        ylabel += r"/Z_" + self.norm + r"^2$"
                    ax[v_idx].set_ylabel(ylabel)
                    # ax[v_idx].set_xlim([0.0, 1.0])
                    self.twist_diffs[sub_idx + 1][momvar_idx + 1][vertex] = {
                        "del_aq4": np.abs(del_aq4),
                        "Z_diff": diff,
                        "label": label,
                    }
                    if zoom:
                        xmin, xmax = ax[v_idx].get_xlim()
                        ax[v_idx].set_xlim([-0.1, 1.0])
                    if subtract_q6:
                        res = fit_func(
                            del_aq4,
                            diff,
                            twist_diff_ansatz,
                            guess=[0.1, 1, 1] if num_params == 3 else [1, 1],
                            del_aq6=del_aq6,
                            num_params=num_params,
                            correlated=False,
                        )
                        chi_sq_red = int(res.chi_sq/res.DOF)
                        fit_label = r'$\chi_\mathrm{red}='+str(chi_sq_red)+r'$'
                        a4_diff = diff - res.val[-1] * del_aq6
                        ax[v_idx].errorbar(
                            np.abs(del_aq4),
                            a4_diff.val,
                            yerr=a4_diff.err,
                            fmt="o",
                            capsize=4,
                            color=pltcolors[momvar_idx],
                            label=label,
                        )
                        self.twist_diffs[sub_idx + 1][momvar_idx + 1][vertex].update(
                            {"del_aq6": del_aq6, "Z_diff_no_aq6": a4_diff}
                        )
                        xmin, xmax = ax[v_idx].get_xlim()
                        xrange = np.linspace(0, xmax, 50)
                        yrange = res[-2] * xrange
                        yrange *= 1 if sign > 0 else -1
                        ax[v_idx].fill_between(
                            xrange,
                            yrange.val + yrange.err,
                            yrange.val - yrange.err,
                            color=pltcolors[momvar_idx],
                            alpha=0.2,
                            label=fit_label,
                        )
                        ax[v_idx].set_xlim([xmin, xmax])

        ax[-1].set_xlabel(r"$\Delta \sum_i(aq_i)^4$")
        handles, labels = ax[-1].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="center",
            ncol=self.N_tw - 1,
            fontsize="small",
            columnspacing=0.5,
            bbox_to_anchor=(0.5, 0.87),
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(1)

        fname = f"plots/{self.ens.name}_fq_Zs_chiral_extrap_tw_diffs.pdf"
        callPDF(fname, show=show)
        print(f"plotted to {os.getcwd()}/{fname}")

        self.aq4 = {key: np.sum(arr**4, axis=1)
                    for key, arr in self.aqvecs.items()}

    def plot_chiral_extrap_allmomvar(self, subscheme: str, show: bool = False) -> None:
        fig, ax = plt.subplots(
            nrows=len(self.decay_vertices), sharex=True, gridspec_kw={"hspace": 0}
        )
        sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
        title = self.scheme + r"$^{" + sublabel + r"}$, $m_\pi=0$, all twists"
        plt.suptitle(title)

        for momvar_idx in range(self.N_tw):
            Zs = self.load_chiral_extrap(momvar_idx, subscheme, plot=False)
            for v_idx, vertex in enumerate(self.decay_vertices):
                ax[v_idx].errorbar(
                    self.momenta,
                    Zs[vertex].val,
                    yerr=Zs[vertex].err,
                    fmt="o",
                    capsize=4,
                    label=self.mom_combos[momvar_idx],
                )
                if momvar_idx == 0:
                    ylabel = r"$Z" + self.decay_vtx_str[v_idx]
                    if self.norm!="q"
                        ylabel += r"/Z_" + self.norm + r"^2$"
                    ax[v_idx].set_ylabel(ylabel)

        ax[-1].set_xlabel(r"$\sqrt{q^2}$ [GeV]")
        handles, labels = ax[-1].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="center",
            ncol=self.N_tw,
            fontsize="small",
            columnspacing=0.5,
            bbox_to_anchor=(0.5, 0.9),
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(1)

        fname = f"plots/{self.ens.name}_fq_Zs_chiral_extrap_all_tw.pdf"
        callPDF(fname, show=show)
        print(f"plotted to {os.getcwd()}/{fname}")

    def load_chiral_extrap(
        self, momvar_idx: int, subscheme: str, plot: bool = False, show: bool = False
    ) -> Dict:

        self.compute = False
        Zs = self.get_Z_all_masses(momvar_idx, subscheme, plot=False)
        self.pion = TwoPointFn(
            self.ens.name, compute=False, scheme=self.scheme)
        self.pion_masses = join_stats(self.pion.load_meson_masses())

        file = h5py.File(self.Zdata_fname, "r")
        grp_name = f"Fourquark/m0p0/{subscheme}" + f"/momvar_{momvar_idx+1}"
        grp = file[grp_name]

        extrap = {
            vertex: Stat(
                val=np.array(grp[f"{vertex}/central"][:]),
                err=np.array(grp[f"{vertex}/errors"][:]),
                btsp=np.array(grp[f"{vertex}/bootstrap"][:]),
            )
            for vertex in self.decay_vertices
        }
        file.close()

        if plot:
            fig, ax = plt.subplots(
                nrows=len(self.decay_vertices), ncols=1, figsize=(3, 5)
            )
            plt.subplots_adjust(hspace=0)
            sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
            title = (
                self.scheme
                + r"$^{"
                + sublabel
                + r"}$, mom combo "
                + str(momvar_idx + 1)
            )
            ax[0].set_title(title)

            for v_idx, vertex in enumerate(self.decay_vertices):
                ax[v_idx].errorbar(
                    self.momenta,
                    extrap[vertex].val,
                    yerr=extrap[vertex].err,
                    fmt="o",
                    capsize=4,
                    label=f"extrap",
                    c="k",
                )
                for m_idx, mass in enumerate(self.masses):
                    pion_label = err_disp(
                        self.pion_masses.val[m_idx], self.pion_masses.err[m_idx]
                    )
                    ax[v_idx].errorbar(
                        self.momenta,
                        Zs[mass][vertex].val,
                        yerr=Zs[mass][vertex].err,
                        fmt="o",
                        capsize=4,
                        label=r"$m_\pi=" + pion_label + r"$",
                    )
                ylabel = r"$Z" + self.decay_vtx_str[v_idx]
                if self.norm!="q"
                    ylabel += r"/Z_" + self.norm + r"^2$"
                ax[v_idx].set_ylabel(ylabel)

            ax[-1].set_xlabel(r"$\sqrt{q^2}$ [GeV]")
            handles, labels = ax[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center right")
            title = (
                self.scheme
                + r"$^{"
                + sublabel
                + r"}$, $m_\pi=0$ mom combo "
                + str(momvar_idx + 1)
            )
            fname = f"plots/{self.ens.name}_fq_Zs_chiral_extrap_tw{momvar_idx}.pdf"
            callPDF(fname, show=show)
            print(f"plotted to {os.getcwd()}/{fname}")

        return extrap

    def plot_chiral_extrap(self, mpis: Stat, Zs: Stat, res: Stat, title: str) -> None:

        fig, ax = plt.subplots()
        x = mpis**2
        ax.errorbar(
            x.val,
            Zs.val,
            xerr=x.err,
            yerr=Zs.err,
            fmt="o",
            capsize=4,
            label=r"$Z(am_q)$",
        )
        ax.errorbar(
            [0.0],
            [res.val[0]],
            yerr=[res.err[0]],
            fmt="o",
            capsize=4,
            label=r"$Z(am_q=0)$",
        )
        ax.axvline(0, color="k", ls="dashed")
        xmin, xmax = ax.get_xlim()
        xrange = np.linspace(-0.05, mpis.val[-1], 50)
        yrange = res.mapping(xrange)
        ax.fill_between(
            xrange**2,
            yrange.val + yrange.err,
            yrange.val - yrange.err,
            color="k",
            alpha=0.2,
            label=r"fit",
        )
        ax.text(
            0.5,
            0.1,
            r"$\chi^2/$DOF$=" + str(np.around(res.chi_sq / res.DOF, 3)) + r"$",
            va="center",
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_xlim([xmin, xmax])
        ax.set_xlabel(r"$m_\pi^2$ [GeV${}^2$]")
        ax.set_ylabel(r"$Z_\Gamma$")
        ax.set_title(title)
        ax.legend()

    def chiral_extrap(
        self, momvar_idx: int, subscheme: str, plot: bool = False, save: bool = True
    ) -> Dict:

        self.compute = False
        Zs = self.get_Z_all_masses(momvar_idx, subscheme, plot=False)
        self.pion = TwoPointFn(self.ens.name, compute=False)

        if np.all(self.masses == self.pion.masses):
            self.pion_masses = join_stats(self.pion.load_meson_masses())
        else:
            print(
                f"mismatch between NPR masses {self.masses} and "
                + "\nvalence masses {self.pion.masses}"
            )
            return None

        extrap = {}

        for vertex in self.decay_vertices:
            extrap[vertex] = []
            for m_idx in tqdm(range(len(self.momenta)), leave=False, desc=vertex):
                mom = self.momenta[m_idx]
                ys = join_stats([Zs[mass][vertex][m_idx]
                                for mass in self.masses])
                res = fit_func(
                    self.pion_masses, ys, chiral_ansatz, [1, 1], correlated=False
                )
                extrap[vertex].append(res[0])
            extrap[vertex] = join_stats(extrap[vertex])

        if save:
            file = h5py.File(self.Zdata_fname, "a")
            grp_name = f"Fourquark/m0p0/{subscheme}" + \
                f"/momvar_{momvar_idx+1}"

            if grp_name in file.keys():
                del file[grp_name]

            grp = file.create_group(grp_name)
            grp.attrs["momentum_variation"] = self.mom_combos[momvar_idx]

            grp.create_dataset("ap", data=np.array(
                self.momenta) / self.ens.ainv)
            grp.create_dataset("aq", data=self.aqvecs[momvar_idx + 1])
            for vertex in self.decay_vertices:
                grp.create_dataset(f"{vertex}/central",
                                   data=extrap[vertex].val)
                grp.create_dataset(f"{vertex}/errors", data=extrap[vertex].err)
                grp.create_dataset(f"{vertex}/bootstrap",
                                   data=extrap[vertex].btsp)

            print(f"saved data to {grp_name} in {self.Zdata_fname}")

        if plot:
            sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
            title = (
                self.scheme
                + r"$^{"
                + sublabel
                + r"}$, $m_\pi=0$ mom combo "
                + str(momvar_idx + 1)
            )
            fname = f"plots/{self.ens.name}_fq_Zs_chiral_extrap_tw{momvar_idx}.pdf"
            self.plot_Z_factors(extrap, title, fname)
        return extrap

    def get_Z_all_masses(
        self,
        momvar_idx: int,
        subscheme: str,
        plot: bool = True,
        run: bool = False,
        **kwargs,
    ) -> Dict:

        Zs = {
            mass: self.get_Z_all_mom(
                mass, momvar_idx, subscheme, run=run, **kwargs)
            for mass in self.masses
        }

        if plot:
            fig, ax = plt.subplots(nrows=len(self.decay_vertices))
            plt.subplots_adjust(hspace=0)
            sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
            title = (
                self.scheme
                + r"$^{"
                + sublabel
                + r"}$, mom combo "
                + str(momvar_idx + 1)
            )
            ax[0].set_title(title)

            for v_idx, vertex in enumerate(self.decay_vertices):
                for m_idx, mass in enumerate(self.masses):
                    ax[v_idx].errorbar(
                        self.momenta,
                        Zs[mass][vertex].val,
                        yerr=Zs[mass][vertex].err,
                        fmt="o",
                        capsize=4,
                        label=f"{np.around(mass, 3)}",
                    )
                ylabel = r"$Z" + self.decay_vtx_str[idx]
                if self.norm!="q"
                    ylabel += r"/Z_" + self.norm + r"^2$"
                ax[idx].set_ylabel(ylabel)

            ax[-1].set_xlabel(r"$\sqrt{q^2}$ [GeV]")
            handles, labels = ax[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center right")

            fname = f"plots/{self.ens.name}_fq_Zs_all_masses_tw{momvar_idx}.pdf"
            callPDF(fname, show=False)
            print(f"plotted to {os.getcwd()}/{fname}")

        return Zs

    def get_Z_all_mom(
        self,
        mass: float,
        momvar_idx: int,
        subscheme: str,
        normalise: bool = True,
        run: bool = True,
        plot: bool = False,
    ) -> Dict:
        Zs = {}

        if self.compute and run:
            for idx in tqdm(
                range(len(self.momenta)), leave=False, desc=str(np.around(mass, 3))
            ):
                mom = self.momenta[idx]
                proj_verts = self.project_vertices(
                    mass, mom, momvar_idx, subscheme)
                for key, vertex in proj_verts.items():
                    if key in Zs:
                        Zs[key].append(vertex ** (-1))
                    else:
                        Zs[key] = [vertex ** (-1)]

            Zs = {key: join_stats(Z) for key, Z in Zs.items()}

            self.save_Z_factors(Zs, mass, momvar_idx, subscheme)

        else:
            Zs = self.load_Z_factors(mass, momvar_idx, subscheme)

        if normalise:
            Z_bl = self.bilinear.get_Z_all_mom(
                mass, momvar_idx, subscheme)[self.norm]
            for vertex, mtx in Zs.copy().items():
                if self.norm == "q":
                    Zs[vertex] = mtx * (Z_bl**2)
                else:
                    Zs[vertex] = mtx / (Z_bl**2)

        if plot:
            sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
            title = (
                self.scheme
                + r"$^{"
                + sublabel
                + r"}$, $am_q="
                + str(np.around(mass, 3))
                + r"$ mom combo "
                + str(momvar_idx + 1)
            )
            fname = f"plots/{self.ens.name}_fq_Zs_{
                self.mass_map[mass]}_tw{momvar_idx}.pdf"
            self.plot_Z_factors(Zs, title, fname)

        return Zs

    def load_Z_factors(self, mass: float, momvar_idx: int, subscheme: str) -> Dict:

        file = h5py.File(self.Zdata_fname, "r")
        grp_name = (
            f"Fourquark/{self.mass_map[mass]
                         }/{subscheme}"
            + f"/momvar_{momvar_idx+1}"
        )

        grp = file[grp_name]
        self.momenta = list(np.array(grp["ap"][:]) * self.ens.ainv)
        self.aqvecs[momvar_idx + 1] = np.array(grp["aq"][:])
        Zs = {}
        for vertex in grp.keys():
            if vertex not in ["ap", "aq"]:
                Zs[vertex] = Stat(
                    val=grp[f"{vertex}/central"][:],
                    err=grp[f"{vertex}/errors"][:],
                    btsp=grp[f"{vertex}/bootstrap"][:],
                )

        file.close()
        return Zs

    def save_Z_factors(
        self, Zs: Dict, mass: float, momvar_idx: int, subscheme: str
    ) -> None:

        file = h5py.File(self.Zdata_fname, "a")
        grp_name = (
            f"Fourquark/{self.mass_map[mass]
                         }/{subscheme}"
            + f"/momvar_{momvar_idx+1}"
        )

        if grp_name in file.keys():
            del file[grp_name]

        grp = file.create_group(grp_name)
        grp.attrs["momentum_variation"] = self.mom_combos[momvar_idx]

        grp.create_dataset("ap", data=np.array(self.momenta) / self.ens.ainv)
        grp.create_dataset("aq", data=self.aqvecs[momvar_idx + 1])

        for vertex in Zs.keys():
            grp.create_dataset(f"{vertex}/central", data=Zs[vertex].val)
            grp.create_dataset(f"{vertex}/errors", data=Zs[vertex].err)
            grp.create_dataset(f"{vertex}/bootstrap", data=Zs[vertex].btsp)

        file.close()
        print(f"saved data to {grp_name} in {self.Zdata_fname}")

    def plot_Z_factors(self, Zs: Dict, title: str, fname: str) -> None:
        fig, ax = plt.subplots(nrows=2, sharex=True)
        plt.subplots_adjust(hspace=0)

        for idx, vertex in enumerate(self.decay_vertices):
            ax[idx].errorbar(
                self.momenta, Zs[vertex].val, yerr=Zs[vertex].err, fmt="o", capsize=4
            )
            ylabel = r"$Z" + self.decay_vtx_str[idx]
            if self.norm!="q"
                ylabel += r"/Z_" + self.norm + r"^2$"
            ax[idx].set_ylabel(ylabel)
        ax[1].set_xlabel(r"$\sqrt{q^2}$ [GeV]")
        ax[0].set_title(title)
        callPDF(fname, show=False)
        print(f"plotted to {os.getcwd()}/{fname}")

    def project_vertices(
        self, mass: float, mom: float, momvar_idx: int, subscheme: str
    ) -> Dict:

        mass_str, mom_str = self.mass_map[mass], self.mom_map[mom]
        theta_in, theta_out = self.theta_str[mass_str][mom_str][momvar_idx]

        amputees = self.amputate_fourquarks(
            mass=mass, mom=mom, theta_in=theta_in, theta_out=theta_out
        )

        mom_in = convert_to_phys(theta_in, self.ens.L, self.ens.T)
        mom_out = convert_to_phys(theta_out, self.ens.L, self.ens.T)
        qvec = mom_in - mom_out if self.scheme == "SMOM" else mom_in

        projectors = fourquark_projectors(subscheme, qvec=qvec)
        projected = {}
        for key, mtx in amputees.items():
            proj = projectors[key]
            projected[key] = Stat(
                val=np.einsum("abcd,badc", proj.val,
                              mtx.val, optimize=True).real,
                btsp=np.array(
                    [
                        np.einsum(
                            "abcd,badc", proj.btsp[k], mtx.btsp[k], optimize=True)
                        for k in range(mtx.N_boot)
                    ]
                ).real,
            )

        return projected

    def amputate_fourquarks(
        self, mass: float, mom: float, theta_in: np.ndarray, theta_out: np.ndarray
    ) -> Dict:

        operators = self.construct_operators(
            mass=mass, mom=mom, theta_in=theta_in, theta_out=theta_out
        )
        in_prop = self.read_in_externalLeg(mass, mom, theta_in)
        out_prop = self.read_in_externalLeg(mass, mom, theta_out)
        out_prop_g5 = out_prop.use_func(g5)

        in_prop_inv = in_prop.use_func(np.linalg.inv)
        out_prop_inv = out_prop_g5.use_func(np.linalg.inv)

        amputees = {}
        for key, mtx in operators.items():
            amputees[key] = Stat(
                val=fourquark_amputation(
                    out_prop_inv.val, in_prop_inv.val, mtx.val),
                btsp=np.array(
                    [
                        fourquark_amputation(
                            out_prop_inv.btsp[k], in_prop_inv.btsp[k], mtx.btsp[k]
                        )
                        for k in range(mtx.N_boot)
                    ]
                ),
            )
        return amputees

    def construct_operators(self, **loadkwargs) -> Dict:
        fqs = self.read_in_fqs(**loadkwargs)

        doubles = {
            "VA": sum(
                [
                    fqs[self.vertices.index(
                        [f"Gamma{mu}", f"Gamma{mu}Gamma5"])]
                    for mu in dirs
                ],
                Zero(fqs[0].shape),
            ),
            "AV": sum(
                [
                    fqs[self.vertices.index(
                        [f"Gamma{mu}Gamma5", f"Gamma{mu}"])]
                    for mu in dirs
                ],
                Zero(fqs[0].shape),
            ),
        }
        operators = {"VApAV": doubles["VA"] + doubles["AV"]}

        for key, mtx in operators.copy().items():
            transposed_mtx = mtx.use_func(np.swapaxes, axis1=1, axis2=3)
            operators[key + "S-"] = (mtx - transposed_mtx) / 2
            operators[key + "S+"] = (mtx + transposed_mtx) / 2

        return operators

    def read_in_fqs(
        self, mass: float, mom: float, theta_in: np.ndarray, theta_out: np.ndarray
    ) -> List:

        theta_in_str = "_".join(theta_in)
        theta_out_str = "_".join(theta_out)

        files = [
            f"{self.path}/{self.mass_map[mass]}/{self.mom_map[mom]}"
            + f"/{self.prefix}{theta_in_str}_{theta_out_str}.{cf}.h5"
            for cf in self.cf_list
        ]

        data = np.empty(
            shape=(self.N_cf, len(self.vertices), N_cd, N_cd, N_cd, N_cd),
            dtype="complex128",
        )

        for cf in range(self.N_cf):
            try:
                file = h5py.File(files[cf], "r")["FourQuarkFullyConnected"]
            except OSError:
                print(files[cf])
                pdb.set_trace()
            for vx in range(len(self.vertices)):
                corr = file[f"FourQuarkFullyConnected_{vx}"]["corr"][0, 0, :]

                data[cf, vx] = (
                    np.array(corr["re"] + 1j * corr["im"])
                    .swapaxes(1, 2)
                    .swapaxes(5, 6)
                    .reshape((N_cd, N_cd, N_cd, N_cd), order="F")
                )

        fourquarks = [
            Stat(
                val=np.mean(data[:, f_idx], axis=0),
                btsp=bootstrap(data[:, f_idx], seed=self.ens.seed),
            )
            for f_idx in range(len(self.vertices))
        ]

        return fourquarks

    def read_in_externalLeg(self, mass: float, mom: float, theta: np.ndarray) -> Stat:
        """Given theta, reads in data for external leg"""

        prefix = "ExternalLeg_0_"
        theta_str = "_".join(theta)

        files = [
            f"{self.path}/{self.mass_map[mass]}/{self.mom_map[mom]}"
            + f"/{prefix}{theta_str}.{cf}.h5"
            for cf in self.cf_list
        ]

        data = np.empty(shape=(self.N_cf, N_cd, N_cd), dtype="complex128")

        for cf in range(self.N_cf):
            try:
                corr = h5py.File(files[cf], "r")[
                    "ExternalLeg"]["corr"][0, 0, :]
            except OSError:
                print(fname)
                pdb.set_trace()

            data[cf] = (
                np.array(corr["re"] + 1j * corr["im"])
                .swapaxes(1, 2)
                .reshape((N_cd, N_cd), order="F")
            )

        externalleg = Stat(
            val=np.mean(data, axis=0), btsp=bootstrap(data, seed=self.ens.seed)
        )

        return externalleg

    def get_fq_list(self, path: str) -> np.ndarray:
        """get the list of fourquark momentum combinations"""

        all_files = [f for f in os.listdir(path) if f.startswith(self.prefix)]
        mom_combinations = []
        for f in all_files:
            config, mom1, mom2 = decode_fname(f)
            if [mom1, mom2] in mom_combinations:
                continue
            else:
                partial_str = f"{self.prefix}" + \
                    "_".join(mom1) + "_" + "_".join(mom2)
                other_configs = [
                    f for f in all_files if f.startswith(partial_str)]
                if len(other_configs) == self.N_cf:
                    mom_combinations.append([mom1, mom2])
                else:
                    print(
                        f"only {len(other_configs)} config files"
                        + f" found for ({mom1}, {mom2}) in {path}\n"
                    )

        if self.scheme == "SMOM":
            return SMOM_combo_sort(np.array(mom_combinations))
        else:
            return MOM_combo_sort(np.array(mom_combinations))

    def create_attributes(self) -> None:
        self.theta_str = {
            mass_str: {
                mom_str: self.get_fq_list(f"{self.path}/{mass_str}/{mom_str}")
                for mom_str in os.listdir(f"{self.path}/{mass_str}")
            }
            for mass, mass_str in self.mass_map.items()
        }

        self.mom_map = {
            np.linalg.norm(convert_to_phys(
                theta[0][0], self.ens.L, self.ens.T))
            * self.ens.ainv: mom_str
            for mom_str, theta in self.theta_str[self.mass_map[self.masses[0]]].items()
        }
        self.momenta = sorted(list(self.mom_map.keys()))

        self.aqvecs = {
            momvar_idx + 1: np.zeros(shape=(len(self.momenta), N_dir))
            for momvar_idx in range(self.N_tw)
        }
        for momvar_idx in range(self.N_tw):
            for mom_idx, mom in enumerate(self.momenta):
                mass_str = self.mass_map[self.masses[0]]
                mom_str = self.mom_map[mom]
                theta_in, theta_out = self.theta_str[mass_str][mom_str][momvar_idx]
                p_in = convert_to_phys(theta_in, self.ens.L, self.ens.T)
                p_out = convert_to_phys(theta_out, self.ens.L, self.ens.T)
                self.aqvecs[momvar_idx + 1][mom_idx, :] = (
                    p_in - p_out if self.scheme == "SMOM" else p_in
                )


def fourquark_projectors(subscheme: str, qvec: np.ndarray) -> Dict:
    if subscheme not in ["gamma", "qslash"]:
        raise Exception("subscheme input is either gamma or qslash (str)")

    myGamma = Gamma.copy()

    if subscheme == "qslash":
        qsl, qsq = qslash(qvec)
        # replace \gamma_\mu with \slashed{q}q_\mu/q^2
        for i in range(N_dir):
            myGamma[dirs[i]] = qsl * qvec[i] / qsq

    VA = np.sum(
        [np.tensordot(myGamma[i], myGamma[i] @ myGamma["5"], axes=0)
         for i in dirs],
        axis=0,
    )
    AV = np.sum(
        [np.tensordot(myGamma[i] @ myGamma["5"], myGamma[i], axes=0)
         for i in dirs],
        axis=0,
    )

    projectors = {"VApAV": VA + AV}

    for key, mtx in projectors.copy().items():
        transposed_mtx = mtx.swapaxes(1, 3)
        projectors[key + "S-"] = (mtx - transposed_mtx) / 2
        projectors[key + "S+"] = (mtx + transposed_mtx) / 2

    tree_values = {
        vtx: np.einsum("abcd,badc", proj, proj, optimize=True)
        for vtx, proj in projectors.items()
    }

    return {
        vtx: Stat(val=proj / tree_values[vtx], btsp="constant")
        for vtx, proj in projectors.items()
    }


def fourquark_amputation(out_, in_, op_):
    """amputates external legs of fourquark Green"s functions"""

    return np.einsum("ea,bf,gc,dh,abcd->efgh", out_, in_, out_, in_, op_, optimize=True)


def twist_diff_ansatz(del_aq4, param, del_aq6=None, num_params=2, **kwargs):
    if not isinstance(del_aq6, np.ndarray):
        raise Exception("del_aq6 not passed")
    if num_params == 3:
        return param[0] + param[1] * del_aq4 + param[2] * del_aq6
    else:
        return param[0] * del_aq4 + param[1] * del_aq6
