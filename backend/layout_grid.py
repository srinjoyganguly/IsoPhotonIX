"""Rasterize a finite editor scene on a documented cell-centre grid, in µm."""
import hashlib
import math

import numpy as np

from .geometry_draft import geometry_bounds, validate_geometry

AXES = 'xyz'
MAX_CELLS = 1_200_000


def layout_spec(config):
    scene = validate_geometry(config['geometry'])
    objects = scene['objects']
    if scene['mode'] != 'draft-3d' or not 1 <= len(objects) <= 128:
        raise ValueError('The finite-layout study requires 1–128 physical solids in a 3D draft.')
    resolution, pml = config['resolution'], config['pml_um']
    materials = {m['id']: m for m in scene.get('materials', [])}
    bounds = [geometry_bounds(obj) for obj in objects]
    for obj in objects:
        model = materials.get(obj.get('material_id'))
        if model and (model['model'] != 'constant-nk' or model['k'] != 0):
            raise ValueError(f"{obj['name']} needs a constant, lossless material for this study.")
        if not 1 <= obj['refractive_index'] <= 5:
            raise ValueError('Layout core indices must be in [1,5].')
        if obj['size']['z'] * resolution < 2:
            raise ValueError(f"{obj['name']} is thinner than two grid cells. Increase thickness or resolution.")
    max_n = max(config['cladding_index'], *(o['refractive_index'] for o in objects))
    if config['wavelength_um'] * resolution / max_n < 10 or pml * resolution < 8:
        raise ValueError('Use at least ten cells per material wavelength and eight per PML.')
    grid = []
    for axis in AXES:
        outer = max(max(abs(b['min'][axis]), abs(b['max'][axis])) for b in bounds)
        size = max(scene['domain'][axis], 2 * (outer + pml + .4))
        grid.append(math.ceil(size * resolution - 1e-10))
    cells = math.prod(grid)
    if cells > MAX_CELLS:
        raise ValueError(f'This layout requires {cells:,} cells. The local limit is {MAX_CELLS:,}; reduce the domain or physical dimensions.')
    domain = {axis: count/resolution for axis,count in zip(AXES,grid)}
    source = config['source_um']
    # Snap to the nearest Ey Yee location: (i,j+1/2,k) in cell coordinates.
    offsets = (0., .5, 0.)
    index = [math.floor(source[a]*resolution + n/2 - offset + .5) for a,n,offset in zip(AXES,grid,offsets)]
    actual = {a: (i+off-n/2)/resolution for a,i,n,off in zip(AXES,index,grid,offsets)}
    for a in AXES:
        if max(abs(source[a]), abs(actual[a])) + pml + 1/resolution >= domain[a]/2:
            raise ValueError('Place the dipole at least one grid cell inside all absorbing layers.')
    courant = .5 if config['solver']=='meep' else .99 / math.sqrt(3)
    steps = math.ceil(config['duration_fs']*.299792458*resolution/courant)
    if steps*cells > 1_200_000_000:
        raise ValueError('The layout grid × time steps exceeds 1.2 billion cell updates.')
    return {'domain':domain, 'grid':grid, 'cells':cells, 'estimated_steps':steps,
            'cell_updates':steps*cells, 'source_grid_index':index, 'source_um':actual,
            'requested_source_um':dict(source), 'pml_um':pml, 'grid_spacing_um':1/resolution,
            'planning_memory_mib':round(cells*1024/2**20),
            'note':'Planning estimate. The finite structure ends inside cladding on every side.'}


def _inside(contour, x, y):
    """Even–odd inclusion with an explicit boundary rule on sampled centres."""
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros_like(inside)
    for a,b in zip(contour, contour[1:]+contour[:1]):
        dx,dy = b['x']-a['x'],b['y']-a['y']
        u,v = x-a['x'],y-a['y']
        tol = 1e-11
        boundary |= ((abs(dx*v-dy*u) <= tol*math.hypot(dx,dy)) &
                     (u*dx+v*dy >= -tol) & (u*dx+v*dy <= dx*dx+dy*dy+tol))
        if dy:
            inside ^= ((a['y']>y)!=(b['y']>y)) & (x < a['x']+(y-a['y'])*dx/dy)
    return inside | boundary


def _footprint(obj, x, y):
    a = math.radians(obj['rotation_deg'])
    x,y = x-obj['position']['x'], y-obj['position']['y']
    u,v = math.cos(a)*x+math.sin(a)*y, -math.sin(a)*x+math.cos(a)*y
    if obj['kind']=='box':
        return (abs(u)<=obj['size']['x']/2+1e-12) & (abs(v)<=obj['size']['y']/2+1e-12)
    if obj['kind'] in ('ring','cylinder'):
        radius = np.hypot(u,v)
        mask = radius <= obj['radius']+1e-12
        if obj['kind']=='ring':
            hole = radius < obj['inner_radius']
            if not np.any(hole):
                raise ValueError(f"{obj['name']} has an unresolved inner opening.")
            mask &= ~hole
        return mask
    mask = _inside(obj['vertices'],u,v)
    for hole in obj['holes']:
        cut = _inside(hole,u,v)
        if not np.any(cut):
            raise ValueError(f"{obj['name']} has a hole with zero sampled cells. Refine the grid.")
        mask &= ~cut
    return mask


def rasterize(config):
    """Return physical permittivity in [x,y,z]. Last solid wins on overlap."""
    spec = layout_spec(config)
    axes = [(np.arange(n)+.5-n/2)/config['resolution'] for n in spec['grid']]
    epsilon = np.full(spec['grid'],config['cladding_index']**2,dtype=np.float32)
    records = []
    for obj in config['geometry']['objects']:
        bounds = geometry_bounds(obj)
        sections = [slice(np.searchsorted(axis,bounds['min'][a]-1e-12),
                          np.searchsorted(axis,bounds['max'][a]+1e-12,side='right')) for a,axis in zip(AXES,axes)]
        x,y = np.meshgrid(axes[0][sections[0]],axes[1][sections[1]],indexing='ij')
        footprint = _footprint(obj,x,y)
        slab = epsilon[tuple(sections)]
        count = int(np.count_nonzero(footprint)*slab.shape[2])
        if count == 0:
            raise ValueError(f"{obj['name']} covers zero cell centres. Refine the grid or enlarge the feature.")
        slab[footprint,:] = obj['refractive_index']**2
        records.append({'id':obj['id'], 'sampled_cells_before_overlaps':count})
    spec.update(objects=records, epsilon_xyz_sha256=hashlib.sha256(epsilon.astype('<f4',copy=False).tobytes()).hexdigest(),
                precedence='Last solid in editor object order wins. A polygon hole removes only that polygon. Visibility is display-only.')
    return spec, axes, epsilon
