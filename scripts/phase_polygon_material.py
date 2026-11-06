"""Fractional coverage of extruded silicon polygons on a Cartesian Yee grid."""
import numpy as np


def _positive_linear_mean(a, b):
    """Mean of max(v,0) over a line segment with endpoint values a and b."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    positive = np.maximum(hi, 0)
    triangle = np.divide(positive**2, 2 * (hi-lo), out=np.zeros_like(hi), where=hi>lo)
    return np.where(lo >= 0, .5 * (lo+hi), triangle)


def polygon_fraction(polygons, xs, ys, dx):
    """Integrate disjoint polygon interiors in cells centred at xs and ys.

    Each directed edge contributes the integral of its linear height clipped
    to the y strip. Vertical edges have zero area in this representation.
    """
    xs, ys = np.asarray(xs), np.asarray(ys)
    area = np.zeros((xs.size, ys.size), dtype=np.float64)
    for polygon in polygons:
        xy = np.asarray(polygon, dtype=np.float64)
        neighbour = np.roll(xy, -1, axis=0)
        orientation = np.sign(np.sum(xy[:,0]*neighbour[:,1] - xy[:,1]*neighbour[:,0]))
        if not orientation:
            raise ValueError('A polygon must enclose finite area.')
        for (x0,y0), (x1,y1) in zip(xy, neighbour):
            if x0 == x1:
                continue
            a = np.maximum(xs-dx/2, min(x0, x1))
            b = np.minimum(xs+dx/2, max(x0, x1))
            active = b > a
            if not np.any(active):
                continue
            slope = (y1-y0)/(x1-x0)
            left = (y0 + (a[active]-x0)*slope)[:,None] - ys[None,:]
            right = (y0 + (b[active]-x0)*slope)[:,None] - ys[None,:]
            height = _positive_linear_mean(left+dx/2,right+dx/2) - _positive_linear_mean(left-dx/2,right-dx/2)
            area[active] -= orientation * np.sign(x1-x0) * (b[active]-a[active])[:,None] * height
    fraction = area / dx**2
    if fraction.min() < -1e-8 or fraction.max() > 1+1e-8:
        raise ValueError('Expected disjoint non-self-intersecting circuit polygons.')
    return np.clip(fraction, 0, 1)


def extruded_yee(model, resolution, axes, polygons):
    """Diagonal interface approximation on each electric component's own dual cell.

    A sidewall uses arithmetic mixing tangent to the wall and harmonic mixing
    normal to it. The normal estimate uses the gradient of area coverage.
    Horizontal top and bottom interfaces are then combined as a second layer.
    """
    dx = 1/resolution
    xs, ys, zs = axes
    clad, core = model['cladding_index']**2, model['core_index']**2
    material = np.empty((3,len(xs),len(ys),len(zs)),dtype=np.float32)
    for axis in range(3):
        # Scalar cell centres are (i+1/2,j+1/2,k+1/2). Ex is at (i+1/2,j,k).
        x = xs if axis == 0 else xs-dx/2
        y = ys if axis == 1 else ys-dx/2
        z = zs if axis == 2 else zs-dx/2
        fxy = polygon_fraction(polygons,x,y,dx)
        fz = np.clip((np.minimum(z+dx/2,model['height']/2)-np.maximum(z-dx/2,-model['height']/2))/dx,0,1)
        parallel = clad+(core-clad)*fxy
        if axis == 2:
            material[axis] = 1/((1-fz)/clad + fz/parallel[:,:,None])
        else:
            gx, gy = np.gradient(fxy,dx)
            norm2 = gx*gx + gy*gy
            normal2 = gx*gx if axis==0 else gy*gy
            normal2 = np.divide(normal2,norm2,out=np.zeros_like(normal2),where=norm2>1e-16)
            normal = 1/((1-fxy)/clad+fxy/core)
            layer = parallel + (normal-parallel)*normal2
            material[axis] = clad + (layer[:,:,None]-clad)*fz
    return material
