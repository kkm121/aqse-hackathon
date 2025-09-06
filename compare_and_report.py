#!/usr/bin/env python3
# compare_and_report.py
# Runs AQSE vs baseline (multi-seed), saves summary + plots.

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
from controller import run_aqse, pauli_list_4q, coeffs_4q

def run_mode(mode, seed, iters, shots, outdir):
    if mode == 'aqse':
        df, summary = run_aqse(pauli_terms=pauli_list_4q, coeffs_dict=coeffs_4q,
                               n_qubits=4, iters=iters, shots_per_eval=shots,
                               trust_threshold=0.07, bootstrap_points=8,
                               surrogate_models=6, rf_estimators=50,
                               gate_error_two_q=0.02, pruning_threshold=0.04,
                               seed=seed)
    else:
        # baseline: force measurements only
        df, summary = run_aqse(pauli_terms=pauli_list_4q, coeffs_dict=coeffs_4q,
                               n_qubits=4, iters=iters, shots_per_eval=shots,
                               trust_threshold=-1.0,
                               bootstrap_points=8,
                               surrogate_models=6, rf_estimators=50,
                               gate_error_two_q=0.02, pruning_threshold=0.04,
                               seed=seed)
    # save trace
    fname = os.path.join(outdir, f"trace_{mode}_seed{seed}.csv")
    df.to_csv(fname, index=False)
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-seeds', type=int, default=3)
    p.add_argument('--iters', type=int, default=40)
    p.add_argument('--shots', type=int, default=512)
    p.add_argument('--outdir', type=str, default='results')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for mode in ('aqse','baseline'):
        for s in range(args.n_seeds):
            seed = 1000 + s
            print(f"Running {mode} seed {seed} ...")
            summary = run_mode(mode, seed, args.iters, args.shots, args.outdir)
            rows.append({
                'mode': mode, 'seed': seed,
                'final_energy': summary['final_energy'],
                'total_shots': summary.get('total_shots_aqse', summary.get('total_shots', None)),
                'shots_saved_pct': summary.get('shots_saved_pct', 0.0)
            })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(args.outdir, 'summary.csv'), index=False)

    # plots
    plt.figure(figsize=(6,4))
    df_all.boxplot(column='final_energy', by='mode')
    plt.title('Final energy by mode'); plt.suptitle(''); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'final_energy_box.png'))

    plt.figure(figsize=(6,4))
    df_all.boxplot(column='total_shots', by='mode')
    plt.title('Total shots by mode'); plt.suptitle(''); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'total_shots_box.png'))

    print("Done. Summary and plots in", args.outdir)

if __name__ == '__main__':
    main()
