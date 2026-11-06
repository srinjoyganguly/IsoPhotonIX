"""Read a declared native electromagnetic field plane at its recorded coordinates."""
from pathlib import Path

import numpy as np


def field_plane(report, archive):
    descriptor = report.get('field_preview')
    if not descriptor and report.get('case') in ('multimode-tm', 'layout-ports-2d'):
        component = 'Hz' if report['case']=='multimode-tm' else 'Ez'
        prefix = '' if component=='Hz' else 'port0-device_'
        sample = '0-device_final_Hz' if component=='Hz' else 'port0-device_final_Ez'
        with np.load(Path(archive), allow_pickle=False) as raw:
            descriptor = {'array':sample,'index':[], 'axis_order':'xy','horizontal_axis':'x','vertical_axis':'y',
                          'horizontal_um':raw[prefix+'x_um'], 'vertical_um':raw[prefix+'y_um'],
                          'component':component,'description':'Final instantaneous field for the first physical device launch.'}
    if not descriptor:
        return None
    horizontal = np.asarray(descriptor['horizontal_um'], dtype=float)
    vertical = np.asarray(descriptor['vertical_um'], dtype=float)
    for axis in (horizontal, vertical):
        if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
            raise ValueError('Field coordinates must be finite, increasing one-dimensional samples.')
    indices = descriptor.get('index', [])
    if any(type(i) is not int or i < 0 for i in indices):
        raise ValueError('Invalid field plane selection.')
    with np.load(Path(archive), allow_pickle=False) as raw:
        values = raw[descriptor['array']][tuple(indices)].real
    if descriptor['axis_order'] == 'xy':
        values = values.T
    elif descriptor['axis_order'] != 'yx':
        raise ValueError('Unknown field plane axis order.')
    if values.shape != (vertical.size, horizontal.size) or not np.isfinite(values).all():
        raise ValueError('The measured field plane and its physical coordinates disagree.')

    # Send original samples at original coordinates. Include both edges even on a large mesh.
    xi = np.linspace(0, horizontal.size - 1, min(160, horizontal.size), dtype=int)
    yi = np.linspace(0, vertical.size - 1, min(160, vertical.size), dtype=int)
    return {
        'description': descriptor['description'], 'component': descriptor['component'],
        'horizontal_axis': descriptor['horizontal_axis'], 'vertical_axis': descriptor['vertical_axis'],
        'horizontal_um': horizontal[xi].tolist(), 'vertical_um': vertical[yi].tolist(),
        'values': values[np.ix_(yi, xi)].tolist(), 'peak_abs': float(np.max(np.abs(values))),
        'source_shape': list(values.shape),
        **{key: descriptor[key] for key in ('wavelength_um','time_um_c') if key in descriptor},
    }
