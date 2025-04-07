from fourquarks import *

scheme = "SMOM"
sub_idx = 0
subscheme = "gamma"


def plot_q4_ratios(ens1: str = "C", ens2: str = "M",
                   zoom: bool = False, ylabel: str = "Z_diff",
                   subscheme="gamma", **kwargs):
    fq = []
    for ens in [ens1, ens2]:
        fq.append(Fourquark(ens, compute=False, scheme=scheme))
        fq[-1].plot_twist_diffs(sub_idx=sub_idx, subscheme=subscheme)

    fig, ax = plt.subplots(nrows=len(Fourquark.decay_vertices), sharex=True)
    plt.subplots_adjust(hspace=0)
    sublabel = r"\gamma_\mu" if subscheme == "gamma" else r"\not{q}"
    title = (
        scheme
        + r"$^{"
        + sublabel
        + r"}$, $m_\pi=0$, twist differences "
        + f"{ens1}/{ens2}"
    )
    plt.suptitle(title)
    for momvar_idx in range(fq[0].N_tw):
        if momvar_idx != sub_idx:
            for v_idx, vertex in enumerate(Fourquark.decay_vertices):
                x1 = fq[0].twist_diffs[sub_idx +
                                       1][momvar_idx + 1][vertex]["del_aq4"]
                y1 = fq[0].twist_diffs[sub_idx +
                                       1][momvar_idx + 1][vertex][ylabel]

                x2 = fq[1].twist_diffs[sub_idx +
                                       1][momvar_idx + 1][vertex]["del_aq4"]
                y2 = fq[1].twist_diffs[sub_idx +
                                       1][momvar_idx + 1][vertex][ylabel]
                if ens1 == "F":
                    x1 = x1[1:]
                    y1 = y1[1:]
                elif ens2 == "F":
                    x2 = x2[1:]
                    y2 = y2[1:]

                if np.allclose(x1, x2):
                    x = x1
                    y = y1 / y2
                else:
                    pdb.set_trace()

                label = fq[0].twist_diffs[sub_idx +
                                          1][momvar_idx + 1][vertex]["label"]
                ax[v_idx].errorbar(
                    x,
                    y.val,
                    yerr=y.err,
                    fmt="o",
                    capsize=4,
                    label=label,
                )
                ax[v_idx].set_ylabel(
                    r"$Z"
                    + Fourquark.decay_vtx_str[v_idx]
                    + r"/Z_"
                    + fq[0].norm
                    + r"^2$"
                )
                # if momvar_idx == (sub_idx+1) % fq[0].N_tw:
                #    aratio = fq[0].ens.ainv/fq[1].ens.ainv
                #    arat_label = r"$(a_{"+ens1+r"}/a_{"+ens2+r"})^4$"
                #    ax[v_idx].axhline(aratio**4, c='k', label=arat_label)
                ax[v_idx].set_ylim([0, 2])
                if zoom:
                    ax[v_idx].set_xlim([-0.1, 1.0])

    ax[-1].set_xlabel(r"$\Delta \sum_i(aq_i)^4$")
    handles, labels = ax[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="center",
        ncol=fq[0].N_tw,
        fontsize="small",
        columnspacing=0.5,
        bbox_to_anchor=(0.5, 0.87),
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1)

    fname = f"plots/{ens1}_over_{ens2}_fq_Zs_chiral_extrap_tw_diffs.pdf"
    callPDF(fname, show=True)
    print(f"plotted to {os.getcwd()}/{fname}")


def plot_qvecs(ens: str, scheme: str = 'SMOM', subscheme: str = "gamma",
               tw_idx: int = 1, **kwargs):
    e = Fourquark(ens, compute=False, scheme=scheme)
    for momvar_idx in range(e.N_tw):
        e.load_chiral_extrap(momvar_idx, subscheme, plot=False)

    aqvecs = e.aqvecs[tw_idx]
    aq2 = np.sum(aqvecs**2, axis=-1)
    aq2_2 = aq2**2
    aq2_3 = aq2**3

    aq4 = np.sum(aqvecs**4, axis=-1)
    aq6 = np.sum(aqvecs**6, axis=-1)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    plt.subplots_adjust(hspace=0)

    ax[0].scatter(aq2, aq4, marker='o', label=r'$a^4\sum_i q_i^4$')
    ax[0].scatter(aq2, aq2_2, marker='o', label=r'$((aq)^2)^2$')
    ax[0].set_ylabel(r'$O(a^4)$')
    ax[1].scatter(aq2, aq6, marker='o', label=r'$a^6\sum_i q_i^6$')
    ax[1].scatter(aq2, aq2_3, marker='o', label=r'$((aq)^2)^3$')
    ax[1].set_ylabel(r'$O(a^6)$')
    ax[1].set_xlabel(r'$(aq)^2$')

    ax[0].legend()
    ax[1].legend()

    fname = f"plots/{ens}_aqvecs.pdf"
    callPDF(fname, show=True)
    print(f"plotted to {os.getcwd()}/{fname}")
