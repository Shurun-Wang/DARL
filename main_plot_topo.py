from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# ======================= 可修改的设置 =======================
DATA_DIR = Path(__file__).resolve().parent / 'results_attributions'
OUTPUT_DIR = DATA_DIR / 'adhd_darl_regions'
SEEDS = [10, 20, 30, 40, 50]
FOLD = 1
NORMALIZE = True
FIGURE_SIZE = (6.8, 3.2)
TITLE_SIZE = 12
CHANNEL_SIZE = 8
COLORMAP = 'RdBu_r'
SHOW_FIGURE = True
PERMUTATIONS = 9999
BOOTSTRAPS = 5000
STATISTICAL_SEED = 2026


REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'Temporal': ['T7', 'T8', 'P7', 'P8'],
    'Parietal': ['P3', 'P4', 'Pz'],
    'Occipital': ['O1', 'O2'],
    'Central': ['C3', 'C4', 'Cz'],
}
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """每个受试者、每个种子保留一条通道向量，并算各区域通道均值。"""
    records, seed_maps = [], []
    reference_channels, reference_subjects = None, None
    for seed in SEEDS:
        filename = f'adhd_darl_adhd_fold_{FOLD}_seed_{seed}.npz'
        with np.load(DATA_DIR / filename, allow_pickle=False) as data:
            channels = data['channel_names'].tolist()
            subjects = [np.unique(data[f'{g}_subject_ids']) for g in ['hc', 'pt']]
            if reference_channels is not None:
                if channels != reference_channels or any(not np.array_equal(a, b) for a, b in zip(subjects, reference_subjects)):
                    raise ValueError(f'种子间通道或受试者不一致：{filename}')
            reference_channels, reference_subjects = channels, subjects
            if set(sum(REGIONS.values(), [])) != set(channels):
                raise ValueError('区域定义没有覆盖全部通道，请检查 REGIONS')

            group_maps = []
            for key, group, names in zip(['hc', 'pt'], ['HC', 'ADHD'], subjects):
                values, ids = data[key], data[f'{key}_subject_ids']
                if not len(names) or not np.isfinite(values).all():
                    raise ValueError(f'空组或非有限归因值：{filename}, {key}')
                subject_maps = []
                for subject in names:
                    vector = values[ids == subject].mean(axis=0)  # 受试者内平均全部片段。
                    if NORMALIZE and vector.sum() > 0:
                        vector = vector / vector.sum()
                    subject_maps.append(vector)
                    failed = int((~data['completeness_pass'][data['subject_ids'] == subject]).sum())
                    for region, electrode_names in REGIONS.items():
                        indices = [channels.index(ch) for ch in electrode_names]
                        records.append(dict(dataset='adhd', configuration='darl_adhd', fold=FOLD,
                            seed=seed, subject_id=str(subject), group=group, region=region,
                            value=float(vector[indices].mean()), n_channels=len(indices),
                            n_windows=int((ids == subject).sum()), ig_failed_windows=failed,
                            normalized=NORMALIZE, source_file=filename))
                group_maps.append(np.mean(subject_maps, axis=0))
            seed_maps.append(group_maps)
            failures = int((~data['completeness_pass']).sum())
            if failures:
                print(f'注意：{filename} 有 {failures} 个积分容限失败片段，仍保留。')
    return pd.DataFrame(records), np.mean(seed_maps, axis=0), reference_subjects, reference_channels


def subject_group_test(hc, patients, rng):
    effect = patients.mean() - hc.mean()
    pooled = np.concatenate([hc, patients])
    exceedances = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(pooled)
        difference = shuffled[len(hc):].mean() - shuffled[:len(hc)].mean()
        exceedances += abs(difference) >= abs(effect) - 1e-12
    p = (exceedances + 1) / (PERMUTATIONS + 1)  # Monte Carlo +1 修正，p 不会变成 0。
    # 95% CI 也重采样受试者，而不是把种子记录当独立样本。
    differences = rng.choice(patients, (BOOTSTRAPS, len(patients))).mean(axis=1) - rng.choice(hc, (BOOTSTRAPS, len(hc))).mean(axis=1)
    lower, upper = np.quantile(differences, [.025, .975])
    return effect, p, lower, upper


def save_region_statistics(records):
    records.to_csv(OUTPUT_DIR / 'region_subject_seed_values.csv', index=False, encoding='utf-8-sig')
    summary = records.groupby(['region', 'group'], sort=False).agg(
        mean=('value', 'mean'), std=('value', 'std'), n_subject_seed_records=('value', 'size'),
        n_subjects=('subject_id', 'nunique'), n_seeds=('seed', 'nunique')).reset_index()
    summary.to_csv(OUTPUT_DIR / 'region_mean_std.csv', index=False, encoding='utf-8-sig')

    subject_means = records.groupby(['region', 'group', 'subject_id'], sort=False).value.mean().reset_index()
    tests = []
    rng = np.random.default_rng(STATISTICAL_SEED)
    for region in REGIONS:
        subset = subject_means[subject_means.region == region]
        hc = subset[subset.group == 'HC'].value.to_numpy()
        patients = subset[subset.group == 'ADHD'].value.to_numpy()
        effect, p, lower, upper = subject_group_test(hc, patients, rng)
        tests.append(dict(region=region, effect_ADHD_minus_HC=effect, ci95_low=lower,
            ci95_high=upper, p_two_sided=p, n_HC_subjects=len(hc), n_ADHD_subjects=len(patients),
            test='subject-level label permutation after five-seed averaging'))
    tests = pd.DataFrame(tests)

    p = tests.p_two_sided.to_numpy()
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(adjusted, 1)
    tests['q_FDR'] = q
    tests.to_csv(OUTPUT_DIR / 'region_group_tests.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))


def plot_maps(maps, subjects, channels):
    info = mne.create_info(channels, sfreq=200, ch_types='eeg')
    info.set_montage('standard_1020')
    vmin, vmax = maps.min(), maps.max()       # 对照与患者共用同一色标。
    if vmin == vmax:
        vmax = vmin + 1e-8
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    for ax, vector, group, names in zip(axes, maps, ['Control', 'ADHD'], subjects):
        image, _ = mne.viz.plot_topomap(vector, info, axes=ax, show=False,
            cmap=COLORMAP, vlim=(vmin, vmax), names=channels, contours=0)
        ax.set_title(f'DARL · {group} (n={len(names)})', fontsize=TITLE_SIZE, pad=4)
        for text in ax.texts:
            text.set_fontsize(CHANNEL_SIZE)
    fig.subplots_adjust(left=.02, right=.84, bottom=.03, top=.88, wspace=.12)
    colorbar = fig.colorbar(image, cax=fig.add_axes([.87, .20, .025, .58]))
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label('Absolute IG share' if NORMALIZE else 'Mean absolute IG', fontsize=10)
    fig.savefig(OUTPUT_DIR / 'adhd_darl_topomap.png', dpi=600)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, maps, subjects, channels = load_data()
    save_region_statistics(records)
    plot_maps(maps, subjects, channels)
    print('输出目录：', OUTPUT_DIR)


if __name__ == '__main__':
    main()
