"""Render the two finite-width vendor comparisons from qualified, hash-checked measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def curves(directory, vendor):
    rows = []
    for mesh in (12, 14):
        path = directory / f'v012-grating3d-{vendor}-r{mesh}.json'
        report = json.loads(path.read_text())
        settings = report['settings']
        if not (report['passed'] and all(report['result']['valid_beam_bins']) and
                settings['geometry'] == vendor and settings['resolution'] == mesh):
            raise ValueError(f'Incomplete or unexpected measurement in {path.name}')
        raw = directory / report['arrays_file']
        if hashlib.sha256(raw.read_bytes()).hexdigest() != report['arrays_sha256']:
            raise ValueError(f'Changed numerical archive for {path.name}')
        ref = next(c for c in report['comparisons'] if c.get('geometry_matched'))
        wavelengths = np.asarray(ref['wavelength_um']) * 1000
        published = 100 * np.asarray(ref.get('published_power', ref.get('published_s21_power')))
        with np.load(raw, allow_pickle=False) as data:
            measured = 100 * abs(data['output_mode'])**2 / data['incident_power']
            np.testing.assert_array_equal(data['valid'], np.ones(len(wavelengths), dtype=bool))
        np.testing.assert_array_equal(wavelengths, 1000 * np.asarray(report['result']['wavelength_um']))
        np.testing.assert_allclose(measured, 100 * np.asarray(report['result']['beam_to_guide']), rtol=1e-10)
        rows.append((wavelengths, published, measured))
    np.testing.assert_array_equal(rows[0][0], rows[1][0])
    np.testing.assert_array_equal(rows[0][1], rows[1][1])
    return rows[0][0], rows[0][1], rows[0][2], rows[1][2]


def plot(directory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), sharex='col', sharey='row',
                                height_ratios=(2.2, 1), layout='constrained')
    colors = ('#465469', '#ae5b10', '#007c78')
    for col, (vendor, title) in enumerate((('lumerical', 'Lumerical · curved grating and fiber'),
                                           ('tidy3d', 'Tidy3D · uniform grating and taper'))):
        wavelength, published, coarse, fine = curves(directory, vendor)
        top, bottom = axes[0, col], axes[1, col]
        top.set_title(title, loc='left', fontsize=11, weight='bold')
        top.plot(wavelength, published, 's--', color=colors[0], label='Published')
        for measured, color, marker, label in ((coarse, colors[1], '^', '12 cells/µm'),
                                               (fine, colors[2], 'o', '14 cells/µm')):
            top.plot(wavelength, measured, marker=marker, color=color, label=label)
            bottom.plot(wavelength, measured - published, marker=marker, color=color)
        top.set_ylim(bottom=0)
        bottom.axhline(0, color=colors[0], linewidth=1)
        bottom.set_xlabel('Wavelength (nm)')
        bottom.set_xticks(wavelength)
        for axis in (top, bottom):
            axis.grid(axis='y', color='#d8dfe3', linewidth=.65)
            axis.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Coupling efficiency (%)')
    axes[1, 0].set_ylabel('Local − published\n(percentage points)')
    upper = axes[0, 0].get_ylim()[1]
    axes[0, 0].set_ylim(0, upper * 1.18)
    axes[0, 0].legend(frameon=False, fontsize=8, ncols=3, loc='upper center')
    figure.savefig(output, dpi=170, facecolor='white')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1]/'docs/verification')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.output)
