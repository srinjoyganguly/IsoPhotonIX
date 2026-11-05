"""Physical taper and uniform lead sections for independently calibrated ports."""
import math
from .taper_geometry import taper_geometry


def port_layout(config):
    layout = taper_geometry({**config,'wavelength_um':config['wavelength_min_um']})
    res,pml = config['resolution'],config['pml_um']
    if min(layout['width_in'],layout['width_out'],layout['height'])*res < 6:
        raise ValueError('Resolve both taper end widths and height with at least six cells for modal measurements.')
    length = layout['x_out']-layout['x_in']
    required = {'x':length+2*(pml+2.25),'y':4.,'z':4.}
    domain = {a: math.ceil(max(layout['domain'][a],v)*res-1e-10)/res for a,v in required.items()}
    s = domain['x']/2-pml-.5
    p = s-1.
    return {**layout,'domain':domain,'source_x':-s,'source_forward_x_um':-s,'source_reverse_x_um':s,
            'port_1_x_um':-p,'port_2_x_um':p,'aperture_y_um':domain['y']-2*pml,
            'aperture_z_um':domain['z']-2*pml,'mode':'Fundamental Ey-like guided channel at each end',
            'reference_planes':'Physical port planes in uniform access guides; taper propagation phase is retained.'}
