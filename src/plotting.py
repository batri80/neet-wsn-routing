"""plotting.py -- figures for the NEET v2 experiment pipeline."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'static': 'tab:blue', 'random': 'tab:orange', 'greedy': 'tab:green',
          'mincost': 'tab:red', 'eqopt': 'black', 'LEACH': 'tab:purple',
          'HEED': 'tab:brown', 'PEGASIS': 'tab:pink'}


def plot_scaling(df, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for strat in df.strategy.unique():
        sub = df[df.strategy == strat].groupby('N').agg(
            AUC_Psi=('AUC_Psi', 'mean'), HND=('HND', 'mean')).reset_index()
        axes[0].plot(sub.N, sub.AUC_Psi, 'o-', label=strat, color=COLORS.get(strat))
        axes[1].plot(sub.N, sub.HND, 'o-', label=strat, color=COLORS.get(strat))
    axes[0].set_xlabel('N'); axes[0].set_ylabel('AUC(Psi)'); axes[0].set_title('Exp A: Scaling — AUC(Psi)')
    axes[1].set_xlabel('N'); axes[1].set_ylabel('HND'); axes[1].set_title('Exp A: Scaling — HND')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_energy_sensitivity(df, outpath):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for strat in df.strategy.unique():
        sub = df[df.strategy == strat].groupby('E0').agg(AUC_PsiN=('AUC_PsiN', 'mean')).reset_index()
        ax.plot(sub.E0, sub.AUC_PsiN, 'o-', label=strat, color=COLORS.get(strat))
    ax.set_xlabel('E0 (J)'); ax.set_ylabel('AUC(Psi_N)')
    ax.set_title('Exp B: Energy Sensitivity')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_topology(df, outpath):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    piv = df.groupby(['sink_topology', 'strategy']).agg(AUC_Psi=('AUC_Psi', 'mean')).reset_index()
    topos = ['center', 'edge', 'corner']
    strategies = list(df.strategy.unique())
    x = np.arange(len(topos)); width = 0.15
    for i, strat in enumerate(strategies):
        vals = [piv[(piv.sink_topology == t) & (piv.strategy == strat)].AUC_Psi.values
                for t in topos]
        vals = [v[0] if len(v) else 0 for v in vals]
        ax.bar(x + i * width, vals, width, label=strat, color=COLORS.get(strat))
    ax.set_xticks(x + width * (len(strategies) - 1) / 2); ax.set_xticklabels(topos)
    ax.set_ylabel('AUC(Psi)'); ax.set_title('Exp C: Sink Topology')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_ablation(df, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ['AUC_Psi', 'AUC_PsiN', 'HND']
    strategies = list(df.strategy.unique())
    for ax, metric in zip(axes, metrics):
        means = [df[df.strategy == s][metric].mean() for s in strategies]
        stds = [df[df.strategy == s][metric].std() for s in strategies]
        ax.bar(strategies, means, yerr=stds, color=[COLORS.get(s) for s in strategies])
        ax.set_title(metric); ax.tick_params(axis='x', rotation=30)
    fig.suptitle('Exp E: Ablation')
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_protocol_comparison(df, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    strategies = list(df.strategy.unique())
    means_psi = [df[df.strategy == s].AUC_Psi.mean() for s in strategies]
    means_hnd = [df[df.strategy == s].HND.mean() for s in strategies]
    axes[0].bar(strategies, means_psi, color=[COLORS.get(s, 'gray') for s in strategies])
    axes[0].set_title('Exp F: AUC(Psi) by protocol'); axes[0].tick_params(axis='x', rotation=30)
    axes[1].bar(strategies, means_hnd, color=[COLORS.get(s, 'gray') for s in strategies])
    axes[1].set_title('Exp F: HND by protocol'); axes[1].tick_params(axis='x', rotation=30)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_time_series(series, outpath, title='Psi(t), R_h(t)'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(series['Psi']); axes[0].set_title('Psi(t)'); axes[0].set_xlabel('round')
    axes[1].plot(series['Rh']); axes[1].axhline(0.82, color='red', ls='--', label='R_c ~ 0.82')
    axes[1].set_title('R_h(t)'); axes[1].set_xlabel('round'); axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
