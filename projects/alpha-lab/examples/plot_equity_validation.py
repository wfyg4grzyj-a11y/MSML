"""Export a frozen strategy's scientific validation chart from local artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation',type=Path,required=True)
    args=parser.parse_args()
    root=args.validation
    data=pd.read_csv(root/'selected-daily.csv',parse_dates=['date'])
    summary=json.loads((root/'summary.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,1,figsize=(11,7),gridspec_kw={'height_ratios':[2,1]},sharex=True)
    wealth=np.cumprod(1+data.net.to_numpy())
    axes[0].plot(data.date,wealth,color='#2563eb',lw=1.6,label='Net strategy wealth')
    axes[0].axhline(1,color='#94a3b8',lw=.7)
    axes[0].set_ylabel('Growth of $1')
    axes[0].set_title('AlphaLab: frozen US equity market-neutral pilot',loc='left',weight='bold',pad=14)
    axes[0].legend(loc='upper left',frameon=False)
    drawdown=wealth/np.maximum.accumulate(np.r_[1.,wealth])[1:]-1
    axes[1].fill_between(data.date,drawdown*100,0,color='#ef4444',alpha=.6)
    axes[1].set_ylabel('Drawdown (%)')
    for ax in axes:
        ax.axvspan(pd.Timestamp('2020-01-01'),pd.Timestamp('2023-01-01'),color='#f59e0b',alpha=.08)
        ax.axvspan(pd.Timestamp('2023-01-01'),pd.Timestamp('2026-01-01'),color='#7c3aed',alpha=.08)
        ax.axvline(pd.Timestamp('2020-01-01'),color='#94a3b8',ls='--',lw=.7)
        ax.axvline(pd.Timestamp('2023-01-01'),color='#94a3b8',ls='--',lw=.7)
        ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(data.date.min(),data.date.max())
    axes[0].text(.20,.92,'Train',transform=axes[0].transAxes,color='#475569')
    axes[0].text(.56,.92,'Validation',transform=axes[0].transAxes,color='#92400e')
    axes[0].text(.80,.92,'Frozen holdout',transform=axes[0].transAxes,color='#6d28d9')
    metrics=summary['holdout']
    caption=(f"{summary['selected']['name']} | Holdout annual return {metrics['annualized_return']:.1%}, "
             f"Sharpe {metrics['sharpe_zero_rf']:.2f}, realized SPY beta {metrics['realized_spy_beta']:.3f}\n"
             "Gross exposure <=1; 10 bps trading; 3% short borrow; cash interest omitted. "
             "Terminal prices and dated security classifications remain unaudited.")
    fig.text(.09,.035,caption,ha='left',va='bottom',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.09,right=.98,top=.92,bottom=.15,hspace=.10)
    fig.savefig(root/'validation.png',dpi=160)
    fig.savefig(root/'validation.pdf')


if __name__=='__main__':main()
