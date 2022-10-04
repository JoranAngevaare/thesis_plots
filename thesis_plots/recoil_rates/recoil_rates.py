import wimprates
from wimprates import StandardHaloModel, v_earth

import numpy as np
import matplotlib.pyplot as plt
import numericalunits as nu

import matplotlib as mpl
import string

_kms = nu.km / nu.s
vs = np.linspace(0, 800 * _kms, 100000)

# StandardHaloModel().v_0, StandardHaloModel().v_esc
# from wimprates import observed_speed_dist
#
# v_0, v_esc = SHM.v_0, SHM.v_esc
# SHM_old = StandardHaloModel(v_0=220 * _kms, v_esc=544 * _kms)
# SHM_new = StandardHaloModel(v_0=238 * _kms, v_esc=528 * _kms)


def labeled_vline(x, text, ytext,
                  textoffset=0,
                  text_kwargs=None,
                  verticalalignment='center',
                  color='k', alpha=1, text_alpha=None,
                  **kwargs):
    if text_kwargs is None:
        text_kwargs = {}
    if text_alpha is None:
        text_alpha = alpha
    plt.axvline(x, color=color, alpha=alpha, **kwargs)
    plt.text(x + textoffset, ytext, text, color=color, alpha=text_alpha, rotation='vertical',
             verticalalignment=verticalalignment,
             **text_kwargs)


def labeled_hline(y, text, xtext,
                  textoffset=0,
                  text_kwargs=None,
                  horizontalalignment='center',
                  text_alpha=None,
                  color='k',
                  alpha=1,
                  **kwargs):
    if text_kwargs is None:
        text_kwargs = {}
    if text_alpha is None:
        text_alpha = alpha
    plt.axhline(y, color=color, alpha=alpha, **kwargs)
    plt.text(xtext, y + textoffset, text, color=color, alpha=text_alpha,
             horizontalalignment=horizontalalignment,
             **text_kwargs)

def vel_dist(vs, v_0, v_esc):
    return StandardHaloModel(v_0=v_0 * _kms, v_esc=v_esc * _kms).velocity_dist(vs, None) * _kms


class Bla:
    sigma_nucleon = 1e-47
    mws = np.array([5, 10, 20, 50, 100, 200])
    targets = ('Si', 'Ar', 'Ge', 'Xe')
    cmap = 'viridis_r'
    _subplot_opts = dict(
        left=0.05,  # the left side of the subplots of the figure
        right=0.95,  # the right side of the subplots of the figure
        bottom=0.05,  # the bottom of the subplots of the figure
        top=0.95,  # the top of the subplots of the figure
        wspace=0.05,  # the amount of width reserved for blank space between subplots
        hspace=0.,  # the amount of height reserved for white space between subplots
    )

    figure_settings = dict(figsize=(8, 6), facecolor='white', )
    text_kwargs = dict(                   bbox=dict(boxstyle="round",
                             alpha=0.5,
                             facecolor='gainsboro', ))
    @staticmethod
    def join_x_axes(ax_dict, merge):
        """Merge axes that are in merge to share the same x-axis"""
        ax_dict[merge[0]].get_shared_x_axes().join(*[ax_dict[k] for k in merge])

    @staticmethod
    def estimate_bounds(mw_array):
        bounds = (
                [np.log10(mw_array[0] / (np.log10(mw_array[1]) / np.log10(mw_array[0])))] +
                list((np.log10(mw_array[:-1]) + np.log10(mw_array[1:])) / 2) +
                [np.log10(1.9 * mw_array[-1] - mw_array[-2])]
        )
        return 10 ** np.array(bounds)

    def plot_recoil_rates(self, targets=None):
        if targets is None:
            targets = self.targets
        fig = plt.figure(**self.figure_settings)
        plt.subplots_adjust(**self._subplot_opts)
        shm = StandardHaloModel(v_0=220 * _kms)
        layout = """"""
        legend_key='l'
        assert len(targets) >= 1, f"should have at least one target, got {targets}"
        for i, target in enumerate(targets):
            le = string.ascii_uppercase[i]
            layout += f"""
            {le}{legend_key}
            {le}{legend_key}
            {le}{legend_key}"""
        axes = fig.subplot_mosaic(layout,
                                  gridspec_kw={'height_ratios': [0.1, 1, 0.1] * len(targets),
                                               'width_ratios': [1, 0.03]})
        n_target=len(targets)
        target_keys = string.ascii_uppercase[:n_target]
        self.join_x_axes(axes,target_keys)
        y_max = 0
        y_min = np.inf
        es = np.logspace(-1, np.log10(200), 1000)
        for ax, label in zip(target_keys, targets):
            plt.sca(axes[ax])
            norm = mpl.colors.LogNorm(vmin=self.mws[0], vmax=self.mws[-1])
            for mw in self.mws:

                xs = wimprates.rate_wimp(es=es * 1000 * nu.eV,
                                         mw=mw * nu.GeV / nu.c0 ** 2,
                                         sigma_nucleon=self.sigma_nucleon * nu.cm ** 2,
                                         interaction='SI',
                                         material=label,
                                         halo_model=shm,
                                         )
                plt.plot(es, xs, c=getattr(plt.cm, self.cmap)(norm(mw)), )
                y_max = max(y_max, np.max(xs))
                y_min = min(y_min, np.max(xs))
            axes[ax].text(1 - 0.025,
                          0.9,
                          f'$\mathrm{{{label}}}$',
                          **self.text_kwargs,
                          transform=axes[ax].transAxes,
                          ha='right',
                          va='top',
                          )

        mpl.colorbar.ColorbarBase(ax=axes[legend_key], norm=norm,
                                  orientation='vertical',
                                  cmap=self.cmap,
                                  boundaries=self.estimate_bounds(self.mws),
                                  ticks=self.mws,
                                  label='$\mathrm{M}_\$\mathrm{\chi}$')

        for k in target_keys[:-1]:
            axes[k].set_xticks([])
        y_max = np.ceil(y_max / (10 ** np.floor(np.log10(y_max)))) * 10 ** (np.floor(np.log10(y_max)))
        y_min = 10 ** np.floor(np.log10(y_min))
        for k in target_keys:
            axes[k].set_ylabel('$\mathrm{E_{nr}}$ $\$\mathrm{[keV]}')
            plt.sca(axes[k])
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim(y_min, y_max)


    def plot_velocities(self, targets=None):
        if targets is None:
            targets = self.targets

        fig = plt.figure(**self.figure_settings)
        plt.subplots_adjust(**self._subplot_opts)
        legend_key ='l'
        layout = """
                 A.
                 .."""
        for le, target in zip(string.ascii_uppercase[1:len(targets) + 1], targets):
            layout += f"""
                 .{legend_key}
                 {le}{legend_key}
                 {le}{legend_key}"""
        axes = fig.subplot_mosaic(layout,
                                  gridspec_kw={
                                      'height_ratios': [2, 0.2] + [0.1, 1, 0.1] * len(targets),
                                      'width_ratios': [1, 0.03]})
        n_target=len(targets)
        target_keys = string.ascii_uppercase[1:n_target+1]
        self.join_x_axes(axes, string.ascii_uppercase[:n_target+1])
        es = np.linspace(0, 5, 100)

        new = vel_dist(vs, v_0=238, v_esc=544)
        plt.sca(axes['A'])
        axes['A'].xaxis.set_ticks_position('both')
        axes['A'].xaxis.set_label_position('top')
        plt.plot(vs / _kms,
                 vel_dist(vs, v_0=220, v_esc=544),
                 label='old', color='b')
        labeled_vline(544, '$v_{esc}$', 0.0001, color='b', ls='--', textoffset=5)
        plt.plot(vs / _kms,
                 new,
                 label='new', color='g')

        plt.fill_between(
            vs / _kms,
            vel_dist(vs, v_0=238 - 1.5, v_esc=528),
            vel_dist(vs, v_0=238 + 1.5, v_esc=528),
            color='g',
            alpha=0.5,
        )
        labeled_vline(528, '$v_{esc}$', 0.0001, color='g', ls='--', text_kwargs=dict(ha='left'),
                      textoffset=-25)
        plt.axvspan(528 - 25, 528 + 24, alpha=0.1)
        plt.fill_between(
            vs / _kms,
            vel_dist(vs, v_0=238, v_esc=528 - 25),
            vel_dist(vs, v_0=238, v_esc=528 + 24),
            color='g',
            alpha=0.5,
        )

        plt.xlabel("Speed (\si{km/s})")
        plt.ylabel("Density \si{(km/s)^{-1}}")

        for ax, label in zip(target_keys, targets):
            plt.sca(axes[ax])
            norm = mpl.colors.LogNorm(vmin=self.mws[0], vmax=self.mws[-1])
            for mw in self.mws:
                xs = wimprates.vmin_elastic(es * 1000 * nu.eV, mw * nu.GeV / nu.c0 ** 2, label) / _kms
                plt.plot(xs, es, c=getattr(plt.cm, self.cmap)(norm(mw)), )
            mpl.colorbar.ColorbarBase(ax=axes[ax.lower()], norm=norm,
                                      orientation='vertical',
                                      cmap=self.cmap,
                                      boundaries=self.estimate_bounds(self.mws),
                                      ticks=self.mws,
                                      label='$M_\chi$')
            axes[ax].text(0.025,
                          0.9,
                          label,
                          bbox=dict(boxstyle='round', fc="lightgray", ec="k"),
                          transform=axes[ax].transAxes,
                          ha='left',
                          va='top',
                          )

        for k in target_keys[:-1]:
            axes[k].set_xticks([])
        for k in target_keys[:-1]:
            axes[k].set_ylabel('$E_{nr}$ [keV]')
            axes[k].set_ylim(es[0], es[-1])
        axes[target_keys[-1]].set_xlim(0, 800)
        axes[target_keys[-1]].set_xlabel('V-min')
