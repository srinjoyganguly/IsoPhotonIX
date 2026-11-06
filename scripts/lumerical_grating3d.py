"""Reconstruct the published curved 3D grating and its tilted step-index fiber."""
import math
import numpy as np


def model(resolution):
    return dict(geometry='lumerical', resolution=resolution, period=.6713, fill=.3992, teeth=38,
                width=15., guide_width=.5, height=.22, etch=.10, radius=25., taper_exponent=1.15,
                target_length=25., output_extension=10., oxide_top=.7, oxide_bottom=1.,
                n_si=math.sqrt(12.085214794079697), n_oxide=math.sqrt(2.0851426458070788),
                fiber_radius=4.5, fiber_cladding_radius=25., fiber_core_n=1.44427, fiber_clad_n=1.43482,
                fiber_center_x=2.74533, angle_deg=10., beam_x=2.8582013154835044,
                source_z=.9901250394579352, sample_z=.82, output_x=-25.5,
                source_half_span=10., upper_half_span=10., pml=.667, max_time_fs=1800.,
                physical_bounds=[[-26.,-10.,-1.2],[16.,10.,1.3]])


def lp01(radius, core, cladding, wavelength):
    """First LP01 root for continuity of a circular step-index fiber's field and derivative."""
    from scipy.optimize import brentq
    from scipy.special import j0, j1, k0, k1
    v=(2*math.pi/wavelength)*radius*math.sqrt(core**2-cladding**2)
    def condition(u):
        w=math.sqrt(v*v-u*u)
        return u*j1(u)/j0(u)-w*k1(w)/k0(w)
    u=brentq(condition,1e-6,min(v,2.40482555769577)-1e-8)
    w=math.sqrt(v*v-u*u)
    neff=math.sqrt(core**2-(u*wavelength/(2*math.pi*radius))**2)
    return u,w,neff


def fiber_radial_field(rho_over_radius, u, w):
    from scipy.special import j0, k0
    r=np.asarray(rho_over_radius)
    result=j0(u*r)
    outer=r>1.
    result[outer]=j0(u)*k0(w*r[outer])/k0(w)
    return result


def xy_coverage(m, xs, ys):
    """Supersample the curved sectors and non-linear taper in the xy plane."""
    tangent=math.asin(m['width']/(2*m['radius']))
    edge=m['radius']*math.cos(tangent)
    length=m['teeth']*m['period']+m['period']*(1-m['fill'])
    base=np.zeros((len(xs),len(ys)),dtype=np.float32)
    top=np.zeros_like(base)
    for x_offset in ((np.arange(4)+.5)/4-.5)/m['resolution']:
        x=(xs+x_offset+m['radius'])[:,None]
        half=m['width']/2+(m['guide_width']-m['width'])/2*(np.clip(edge-x,0.,edge)/edge)**m['taper_exponent']
        for y_offset in ((np.arange(4)+.5)/4-.5)/m['resolution']:
            y=(ys+y_offset)[None,:]
            r=np.hypot(x,y)
            angle=np.arctan2(abs(y),x)
            guide=(x<0)&(abs(y)<m['guide_width']/2)
            taper=(x>=0)&(x<edge)&(abs(y)<half)
            sector=angle<=tangent
            entry=sector&(r>=edge)&(r<m['radius'])
            teeth=sector&(r>=m['radius'])&(r<m['radius']+length)
            # First radial interval is an etch, then an unetched ridge.
            pitch_phase=(r-m['radius'])/m['period']
            ridge=(pitch_phase<m['teeth'])&((pitch_phase%1.)>=(1.-m['fill']))
            output=sector&(r>=m['radius']+length)&(r<m['radius']+length+m['output_extension'])
            solid=guide|taper|entry|output
            base+=(solid|teeth)/16.
            top+=(solid|(teeth&ridge))/16.
    return base,top


def epsilon(m, axes, reference):
    xs,ys,zs=axes
    dx=1/m['resolution']
    eps=np.empty((len(xs),len(ys),len(zs)),np.float32)
    theta=np.deg2rad(m['angle_deg'])
    if not reference:
        base,top=xy_coverage(m,xs,ys)

    def slab(z,low,high):
        return np.clip((min(z+dx/2,high)-max(z-dx/2,low))/dx,0.,1.)

    for k,z in enumerate(zs):
        # The physical fiber is polished by the horizontal TOX plane. For the separate reference
        # it extends down through the domain at the same tilt, with no chip or oxide interface.
        rho2=((xs[:,None]-m['fiber_center_x']-z*np.tan(theta))*np.cos(theta))**2+ys[None,:]**2
        fiber=np.where(rho2<m['fiber_cladding_radius']**2,m['fiber_clad_n']**2,1.)
        fiber=np.where(rho2<m['fiber_radius']**2,m['fiber_core_n']**2,fiber)
        if reference:
            eps[:,:,k]=fiber
            continue
        frac=slab(z,m['oxide_top'],1.e4)
        substrate=slab(z,-1.e4,-m['oxide_bottom'])
        ep=(1.-frac)*m['n_oxide']**2+frac*fiber+substrate*(m['n_si']**2-m['n_oxide']**2)
        etched=slab(z,0.,m['height']-m['etch'])
        full=slab(z,m['height']-m['etch'],m['height'])
        eps[:,:,k]=ep+(etched*base+full*top)*(m['n_si']**2-m['n_oxide']**2)
    return eps


def reflected_fiber_amplitude(scattered, incoming, incident_power, dx):
    # The reciprocal upward basis on the same horizontal plane is (E*,−H*).
    # Its flux equals the absolute incident downward flux. The result is a complex S11 estimate.
    ex,ey,_,hx,hy,_=scattered.transpose(1,0,2,3)
    iex,iey,_,ihx,ihy,_=incoming.transpose(1,0,2,3)
    product=-ex*ihy+ey*ihx+iex*hy-iey*hx
    return product.sum(axis=(-1,-2))*dx**2/(4*incident_power)
