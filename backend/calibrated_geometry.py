"""Bounded silicon photonic test circuits for the independently excited modal ports.

Lengths are micrometres. Polygon vertices are counterclockwise in physical XY.
The same polygons feed native Meep prisms and FDTDX material masks.
"""
from math import cos, pi

CORE_WIDTH = 0.5
CORE_HEIGHT = 0.25
OUTER_OFFSET = 1.25
INNER_OFFSET = (CORE_WIDTH + 0.125) / 2


def _rail(xs, ys):
    half = CORE_WIDTH / 2
    return [(x, y - half) for x, y in zip(xs, ys)] + [
        (x, y + half) for x, y in reversed(list(zip(xs, ys)))]


def _rise(t):
    return (1 - cos(pi * min(1., max(0., t)))) / 2


def multiport_model(case, pml_um=.75, aperture_y_um=2.):
    """Return a physical study. PML enlargement leaves the interior and ports fixed."""
    if case not in ('coupler', 'splitter'):
        raise ValueError('Choose coupler or splitter.')
    if not .5 <= pml_um <= 1.5 or not 1.5 <= aperture_y_um <= 2.4:
        raise ValueError('PML must be 0.5–1.5 µm; port width must be 1.5–2.4 µm.')
    extent = 8.25
    left = [-1.25, 1.25] if case == 'coupler' else [0.]
    ports = []
    for sign, offsets, side in [(1, left, 'left'), (-1, [-1.25, 1.25], 'right')]:
        for y in offsets:
            lane = 'center' if y == 0 else 'lower' if y < 0 else 'upper'
            ports.append({'name': f'{side}_{lane}', 'x': -sign * 4.75, 'y': y,
                          'source_x': -sign * 5.5, 'inward_sign': sign})
    if case == 'coupler':
        xs = [-extent] + [i / 16 for i in range(-64, 65)] + [extent]
        offsets = [INNER_OFFSET + (OUTER_OFFSET - INNER_OFFSET) * _rise((abs(x) - 1.) / 3.)
                   for x in xs]
        polygons = [_rail(xs, [s * y for y in offsets]) for s in (-1, 1)]
    else:
        # One trunk with a continuous merged body where both branches share material.
        # A single outline avoids coincident polygons at the junction. The notch
        # begins where centreline displacement is exactly one half-width.
        from math import acos
        x_notch = -2. + 4. * acos(1. - CORE_WIDTH / OUTER_OFFSET) / pi
        xs = [-extent] + sorted({i / 16 for i in range(-32, 33)} | {x_notch}) + [extent]
        offsets = [OUTER_OFFSET * _rise((x + 2.) / 4.) for x in xs]
        half = CORE_WIDTH / 2
        arms = [(x, max(0., y - half)) for x, y in zip(xs, offsets) if x > x_notch]
        polygon = [(x, -y - half) for x, y in zip(xs, offsets)]
        polygon += [(x, -y) for x, y in reversed(arms)] + [(x_notch, 0.)] + arms
        polygon += [(x, y + half) for x, y in reversed(list(zip(xs, offsets)))]
        polygons = [polygon]
    return {'case': case, 'cell': [12.5 + 2*pml_um, 5.5 + 2*pml_um, 2.5 + 2*pml_um],
            'core_index': 3.48, 'cladding_index': 1.45, 'height': CORE_HEIGHT, 'width': CORE_WIDTH,
            'pml_um': pml_um, 'aperture': [aperture_y_um, 2.5], 'ports': ports, 'polygons': polygons,
            'frequency_per_um': [1/1.55 - .02, 1/1.55, 1/1.55 + .02], 'pulse_fwidth': .16,
            'pulse_cutoff': 6., 'straight_extent': extent}


def straight_polygon(model, launch_index):
    y = model['ports'][launch_index]['y']
    return _rail([-model['straight_extent'], model['straight_extent']], [y, y])
