import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import mne

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 


def get_args():
    parser = argparse.ArgumentParser(description="Load saved attributions and plot topomaps.")
    parser.add_argument('--dataset', default='idd', type=str, choices=['adhd', 'mdd', 'sch', 'idd'])
    parser.add_argument('--model_name', type=str, default='OhCNN',
                        choices=['OhCNN', 'DeprNet', 'SzHNN', 'MBSzEEGNet', 'STGEFormer'])
    return parser.parse_args()


def plot_global_topomaps(hc_orig, pt_orig, hc_darl, pt_darl, ch_names, model_name, dataset, cur_path):
    def process_and_mean(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.nanmean(arr, axis=0) if len(arr) > 0 else np.zeros(len(ch_names))

    hc_mean_orig = process_and_mean(hc_orig)
    pt_mean_orig = process_and_mean(pt_orig)
    hc_mean_darl = process_and_mean(hc_darl)
    pt_mean_darl = process_and_mean(pt_darl)

    info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')
    try:
        info.set_montage('standard_1020')
    except Exception as e:
        print(f"error: {e}")

    vmax_orig = max(np.max(hc_mean_orig), np.max(pt_mean_orig))
    vmin_orig = min(np.min(hc_mean_orig), np.min(pt_mean_orig))

    vmax_darl = max(np.max(hc_mean_darl), np.max(pt_mean_darl))
    vmin_darl = min(np.min(hc_mean_darl), np.min(pt_mean_darl))


    vmax_total = max(vmax_orig, vmax_darl)
    vmin_total = min(vmin_orig, vmin_darl)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    def draw_topo(mean_data, ax, title, vmin, vmax):
        im, _ = mne.viz.plot_topomap(
            mean_data, info, axes=ax, show=False, cmap='RdBu_r', vlim=(vmin, vmax), names=ch_names
        )
        ax.set_title(title, fontsize=15, fontweight='bold', fontfamily='Times New Roman')
        
        for text in ax.texts:
            text.set_fontsize(13)
            text.set_fontweight('bold')
            text.set_fontfamily('Times New Roman')
        return im

    plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.1)

    im_orig = draw_topo(hc_mean_orig, axes[0, 0], f"Expert - HC (n={len(hc_orig)})", vmin_total, vmax_total)
    draw_topo(pt_mean_orig, axes[0, 1], f"Expert - {dataset.upper()} (n={len(pt_orig)})", vmin_total, vmax_total)

    cbar_ax_orig = fig.add_axes([0.4, 0.50, 0.2, 0.02])
    cbar_orig = fig.colorbar(im_orig, cax=cbar_ax_orig, orientation='horizontal')
    cbar_orig.ax.tick_params(labelsize=13)
    for tick in cbar_orig.ax.get_xticklabels(): 
        tick.set_fontfamily('Times New Roman')

    im_darl = draw_topo(hc_mean_darl, axes[1, 0], f"DARL - HC (n={len(hc_darl)})", vmin_total, vmax_total)
    draw_topo(pt_mean_darl, axes[1, 1], f"DARL - {dataset.upper()} (n={len(pt_darl)})", vmin_total, vmax_total)

    cbar_ax_darl = fig.add_axes([0.4, 0.08, 0.2, 0.02])
    cbar_darl = fig.colorbar(im_darl, cax=cbar_ax_darl, orientation='horizontal')
    cbar_darl.ax.tick_params(labelsize=13)
    for tick in cbar_darl.ax.get_xticklabels(): 
        tick.set_fontfamily('Times New Roman')

    plt.suptitle(f"{model_name} Global Spatial Attention Comparison", fontsize=18, fontweight='bold',
                 fontfamily='Times New Roman', y=0.98)

    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    path = cur_path
    save_file = f"{path}/{dataset}_{model_name}_comparison_separated.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_file}")
    plt.close()


def main():
    args = get_args()
    cur_path = os.path.dirname(os.path.abspath(__file__))

    data_10_orig = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_original_seed_10.npz')
    data_20_orig = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_original_seed_20.npz')
    data_30_orig = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_original_seed_30.npz')
    data_40_orig = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_original_seed_40.npz')
    data_50_orig = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_original_seed_50.npz')

    all_hc_original = np.concatenate([
        data_10_orig['hc'], data_20_orig['hc'], data_30_orig['hc'], data_40_orig['hc'], data_50_orig['hc']
    ])
    all_pt_original = np.concatenate([
        data_10_orig['pt'], data_20_orig['pt'], data_30_orig['pt'], data_40_orig['pt'], data_50_orig['pt']
    ])

    data_10_darl = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_darl_seed_10.npz')
    data_20_darl = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_darl_seed_20.npz')
    data_30_darl = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_darl_seed_30.npz')
    data_40_darl = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_darl_seed_40.npz')
    data_50_darl = np.load(cur_path + '/results/' + args.model_name + '/' + args.dataset + '_darl_seed_50.npz')

    all_hc_darl = np.concatenate([
        data_10_darl['hc'], data_20_darl['hc'], data_30_darl['hc'], data_40_darl['hc'], data_50_darl['hc']
    ])
    all_pt_darl = np.concatenate([
        data_10_darl['pt'], data_20_darl['pt'], data_30_darl['pt'], data_40_darl['pt'], data_50_darl['pt']
    ])

    if args.dataset == 'adhd':
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                    'Fz', 'Cz', 'Pz']
    elif args.dataset == 'mdd':
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    elif args.dataset == 'sch':
        ch_names = ['Fp2', 'F8', 'T4', 'T6', 'O2', 'Fp1', 'F7', 'T3', 'T5', 'O1',
                    'F4', 'C4', 'P4', 'F3', 'C3', 'P3', 'Fz', 'Cz', 'Pz']
    else:
        ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    plot_global_topomaps(
        all_hc_original, all_pt_original, all_hc_darl, all_pt_darl,
        ch_names, args.model_name, args.dataset, cur_path
    )


if __name__ == "__main__":
    main()
