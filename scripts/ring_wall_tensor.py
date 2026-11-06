"""Nondispersive sidewall subpixel control for the finite-height loaded-ring benchmark."""
import numpy as np


def in_plane_inverse(fraction, normal_x, normal_y, slab, below, silicon, oxide, background):
    """Return xx, xy, yy of inverse epsilon for a locally flat vertical wall.

    After normal/tangent averaging in the active slab, parallel-average the
    oxide and background along z for in-plane E. Invert in the wall frame,
    then rotate into the global frame. The x/y rows have different Yee centers.
    """
    other = below * oxide + (1 - below - slab) * background
    tangent = slab * (fraction * silicon + (1 - fraction) * background) + other
    normal = slab / (fraction / silicon + (1 - fraction) / background) + other
    inverse_tangent = 1 / tangent
    contrast = 1 / normal - inverse_tangent
    return (inverse_tangent + contrast * normal_x**2,
            contrast * normal_x * normal_y,
            inverse_tangent + contrast * normal_y**2)


def wall_inverse(g, reference, disk_fractions, lateral_fraction, permittivity):
    """Full row-major inverse tensor at FDTDX's E locations; zero xy at port planes."""
    dx = 1 / g['resolution']
    nx, ny, nz = g['shape']
    radius, width, height, bus = (g[k] for k in ('radius_um','width_um','height_um','bus_y_um'))
    silicon, oxide, background = (g[k]**2 for k in ('silicon_index','substrate_index','top_index'))
    inv = np.zeros((9, nx, ny, nz), np.float32)
    # Keep the existing vertical Ez mixture at the staggered Ez locations.
    exact = permittivity(dict(g, ring_interface='exact-area'), reference)
    inv[8] = 1 / exact[2]
    del exact
    zs = (np.arange(nz) - nz / 2) * dx
    bottom, top = zs - dx/2, zs + dx/2
    slab = np.clip(np.minimum(top, height/2) - np.maximum(bottom, -height/2), 0, dx) / dx
    below = np.clip((-height/2 - bottom) / dx, 0, 1)
    for component in (0, 1):
        xs = (np.arange(nx) - nx/2 + .5*(component == 0)) * dx
        ys = (np.arange(ny) - ny/2 + .5*(component == 1)) * dx
        straight = lateral_fraction(xs, ys, dx, radius, width, bus, True, 'exact-area')
        fraction = straight
        normal_x = np.zeros((nx, ny))
        normal_y = np.ones((nx, ny))
        if not reference:
            ring = disk_fractions(xs, ys, dx, radius+width/2) - disk_fractions(xs, ys, dx, radius-width/2)
            # Exact disk integrals can leave machine roundoff at empty or full cells.
            ring = np.where(ring < 1e-10, 0, np.where(ring > 1-1e-10, 1, ring))
            if np.any((straight > 1e-10) & (ring > 0)):
                raise ValueError('Multiple separated sidewalls occupy one Yee cell on this mesh.')
            fraction = np.clip(straight + ring, 0, 1)
            curved = (ring > 0) & (ring < 1)
            radius_at_cell = np.hypot(xs[:, None], ys[None, :])
            normal_x[curved] = (xs[:, None] / np.where(radius_at_cell == 0, 1, radius_at_cell))[curved]
            normal_y[curved] = (ys[None, :] / np.where(radius_at_cell == 0, 1, radius_at_cell))[curved]
        for k in range(nz):
            xx, xy, yy = in_plane_inverse(fraction, normal_x, normal_y, slab[k], below[k], silicon, oxide, background)
            if component == 0:
                inv[0, :, :, k], inv[1, :, :, k] = xx, xy
            else:
                inv[3, :, :, k], inv[4, :, :, k] = xy, yy
    return inv


def forward_lossless_xy(state, config, objects):
    """Specialize the native FDTDX 0.6.2 update for a lossless xy electric block.

    The native B=c*solve(I,epsilon_inverse) reduces to c*epsilon_inverse and
    A=I when conductivity is absent. Keep both staggered off-diagonal updates,
    the native curl/CPML memories, source injection, post-boundary operation
    and full native H step. Avoid zero xz/yz terms and voxelwise 3x3 solves.
    """
    import jax
    import jax.numpy as jnp
    from fdtdx.core.physics.curl import curl_H
    from fdtdx.fdtd.misc import avg_anisotropic_E_component
    from fdtdx.fdtd.update import apply_boundary_post_E_update, pad_fields_for_boundaries, update_H

    index, arrays = state
    inv = arrays.inv_permittivities
    if inv.shape[0] != 9 or arrays.electric_conductivity is not None or arrays.dispersive_P_curr is not None:
        raise ValueError('The wall update requires nine lossless, nondispersive xy-block coefficients.')
    padded_h = pad_fields_for_boundaries(arrays.fields.H, objects, config)
    curl, psi = curl_H(config, padded_h, arrays.fields.psi_E, arrays.alpha, arrays.kappa, arrays.sigma, True)
    padded_curl = pad_fields_for_boundaries(curl, objects, config)
    y_at_x = avg_anisotropic_E_component(padded_curl, component=1, location=0)
    x_at_y = avg_anisotropic_E_component(padded_curl, component=0, location=1)
    c = config.courant_number
    increment = jnp.stack(((c*inv[0])*curl[0] + (c*inv[1])*y_at_x,
                           (c*inv[3])*x_at_y + (c*inv[4])*curl[1], (c*inv[8])*curl[2]))
    electric = arrays.fields.E + increment
    for source in objects.sources:
        def add_source(field):
            return source.update_E(E=field, inv_permittivities=inv, inv_permeabilities=arrays.inv_permeabilities,
                                   time_step=source.adjust_time_step_by_on_off(index), inverse=False)
        electric = jax.lax.cond(source.is_on_at_time_step(index), add_source, lambda field:field, electric)
    arrays = arrays.aset('fields->psi_E',psi).aset('fields->E',apply_boundary_post_E_update(electric,objects))
    arrays = update_H(time_step=index,arrays=arrays,objects=objects,config=config,simulate_boundaries=True)
    return index+1, arrays
