"""Original, bounded photonic geometry recipes; no simulated performance data.

Recipes retain their parameters, material assignment and named geometric ports.
Coordinates are micrometres. Curves are explicitly sampled polygon geometry.
Every generated version-2 document passes the shared draft validator.
"""
from __future__ import annotations

from copy import deepcopy
import math

GENERATOR_VERSION = 1
PIC_REFERENCE = 'https://www.flexcompute.com/tidy3d/examples/notebooks/PICComponents/'
MATERIAL_REFERENCE = 'https://www.flexcompute.com/tidy3d/examples/notebooks/ModeSolverFemwellBenchmark/'


def _material(identifier, name, n, source, notes):
    return dict(id=identifier, name=name, model='constant-nk', reference_wavelength_um=1.55,
                n=n, k=0., source=source, notes=notes)


MATERIALS = [
    _material('generic-core', 'Generic dielectric · n = 2', 2., 'Original workbench design preset.',
              'Illustrative lossless constant index, not a measured material dataset.'),
    _material('air', 'Air approximation · n = 1', 1., 'Vacuum-like design approximation.',
              'Constant n = 1; atmospheric dispersion and absorption are omitted.'),
    _material('silica-reference', 'Silica approximation · n = 1.44', 1.44, MATERIAL_REFERENCE,
              'Constant cladding index used in a public telecom strip benchmark; not a broadband silica model.'),
    _material('silicon-nitride-reference', 'Silicon nitride approximation · n = 2', 2.,
              'Original nominal design placeholder; material data: https://meep.readthedocs.io/en/latest/Materials/',
              'Illustrative nominal index only. Deposition, wavelength and temperature dependence require measured data.'),
    _material('silicon-reference', 'Silicon approximation · n = 3.48', 3.48, MATERIAL_REFERENCE,
              'Constant core index used in a public 1.55 µm strip benchmark; no dispersion or absorption model.'),
    _material('iii-v-reference', 'III–V reference · n = 3.2', 3.2,
              'Original nominal geometry placeholder; not semiconductor material data.',
              'No alloy composition, gain, quantum wells or carrier physics. A passive geometric placeholder, not a laser material.'),
]


def _p(key, label, default, low, high, step=.1, unit='µm'):
    return dict(key=key, label=label, unit=unit, min=low, max=high, step=step, default=default)


def _width(default=.5, label='Core width'):
    return _p('width_um', label, default, .05, 3., .05)


def _height():
    return _p('height_um', 'Finite core height', .22, .02, 2., .02)


def _length(default=8., low=1., high=60., label='Length'):
    return _p('length_um', label, default, low, high, .5)


def _gap():
    return _p('gap_um', 'Edge-to-edge gap', .2, .02, 3., .05)


def _radius(default=2.):
    return _p('radius_um', 'Centreline bend radius', default, .3, 30., .25)


def _recipe(identifier, name, category, summary, parameters, limitations=()):
    return dict(id=identifier, name=name, category=category, summary=summary,
                parameters=parameters, defaults={p['key']:p['default'] for p in parameters},
                limitations=['Editable design geometry. No electromagnetic performance is supplied.',
                             'Named ports are geometric intent, not measured or solved eigenmodes.', *limitations],
                simulation_support='bounded-taper-fields' if identifier == 'taper' else 'design-only')


_RECIPES = [
    _recipe('strip', 'Strip waveguide', 'routing', 'A finite rectangular dielectric guide.', [_length(), _width(), _height()]),
    _recipe('slot', 'Slot waveguide', 'routing', 'Two separated dielectric rails with an editable slot.', [_length(), _width(.3, 'Rail width'), _gap(), _height()]),
    _recipe('rib', 'Rib waveguide', 'routing', 'A raised ridge over a continuous partial-etch slab.', [_length(), _width(), _height(), _p('slab_width_um','Slab width',2.5,.2,10.), _p('slab_height_um','Slab thickness',.09,.01,1.,.01)]),
    _recipe('taper', 'Linear taper', 'routing', 'A polygonal transition between two guide widths.', [_length(6.), _p('input_width_um','Input width',.4,.05,3.,.05), _p('output_width_um','Output width',1.2,.05,5.,.05), _height()]),
    _recipe('bend', 'Circular bend', 'routing', 'A sampled constant-width circular arc.', [_radius(3.), _p('angle_deg','Turn angle',90.,10.,170.,5.,'°'), _width(), _height()], ['Circular arc, not an Euler bend; 48 angular intervals per arc.']),
    _recipe('s_bend', 'Cosine S-bend', 'routing', 'Smooth endpoint tangents join laterally displaced ports.', [_length(8.,2.), _p('offset_um','Lateral offset',2.,.1,12.,.25), _width(), _height()], ['Centreline cosine path sampled at 64 intervals; no bend-loss guarantee.']),
    _recipe('directional_coupler', 'Directional coupler', 'coupling', 'Two parallel guide sections with four named ends.', [_length(8.), _width(), _gap(), _height()], ['Coupling ratio is not inferred from separation or length.']),
    _recipe('y_branch', 'Y branch', 'coupling', 'One input connects to two symmetric curved branches.', [_length(10.,3.), _p('pitch_um','Output centre spacing',2.,.2,10.,.2), _width(), _height()], ['Same-material branch overlap defines the junction. No splitting-ratio claim.']),
    _recipe('mmi_1x2', '1×2 MMI splitter', 'coupling', 'A multimode body with one input and two access tapers.', [_length(6.,1.,30.,'MMI body length'), _p('body_width_um','MMI body width',3.,.5,12.), _p('pitch_um','Output centre spacing',1.5,.2,8.), _p('access_um','Access taper length',2.,.2,8.), _width(), _height()]),
    _recipe('mmi_2x2', '2×2 MMI coupler', 'coupling', 'A multimode body with two inputs and two outputs.', [_length(6.,1.,30.,'MMI body length'), _p('body_width_um','MMI body width',3.,.5,12.), _p('pitch_um','Port centre spacing',1.5,.2,8.), _p('access_um','Access taper length',2.,.2,8.), _width(), _height()]),
    _recipe('crossing', 'Tapered crossing', 'coupling', 'Two orthogonal guides widen through their intersection.', [_length(8.,2.), _p('centre_width_um','Central width',1.2,.1,5.), _width(), _height()], ['Same-material overlap defines the crossing; no crosstalk or insertion-loss claim.']),
    _recipe('ring_bus', 'Ring and bus', 'resonators', 'A ring alongside a straight access guide.', [_radius(), _length(10.,2.,80.,'Bus length'), _width(), _gap(), _height()]),
    _recipe('ring_add_drop', 'Add/drop ring', 'resonators', 'A ring between two independently named access buses.', [_radius(), _length(10.,2.,80.,'Bus length'), _width(), _gap(), _height()]),
    _recipe('racetrack', 'Racetrack resonator', 'resonators', 'A closed capsule-shaped guide and adjacent bus.', [_radius(), _p('straight_um','Racetrack straight length',4.,.2,30.), _length(12.,2.,100.,'Bus length'), _width(), _gap(), _height()], ['Curved boundaries use 32 intervals per semicircle.']),
    _recipe('mzi', 'Mach–Zehnder layout', 'interferometers', 'Symmetric split and merge paths form two interferometer arms.', [_length(16.,4.,80.,'End-to-end length'), _p('separation_um','Arm centre spacing',3.,.2,12.), _p('transition_um','Branch transition length',4.,.5,20.), _width(), _height()], ['Passive balanced layout only; no heater, electrical modulation or calibrated phase shift.']),
    _recipe('bragg', 'Waveguide Bragg grating', 'periodic', 'A continuous guide with periodic width corrugations.', [_p('period_um','Period',.4,.1,2.,.02), _p('periods','Number of periods',16,2,64,1,''), _p('duty','Duty cycle',.5,.1,.9,.05,''), _p('corrugation_um','Side corrugation',.15,.01,1.,.02), _width(), _height()], ['No stop-band or reflectivity is inferred from the drawn period.']),
    _recipe('grating_coupler', 'Linear grating coupler', 'periodic', 'An access taper feeds a finite, partially etched grating aperture.', [_p('period_um','Period',.6,.15,2.,.02), _p('periods','Number of teeth',16,3,64,1,''), _p('duty','Duty cycle',.5,.1,.9,.05,''), _p('aperture_um','Aperture width',4.,1.,20.,.25), _p('taper_um','Access taper length',4.,.5,30.), _p('slab_height_um','Residual slab thickness',.08,.01,1.,.01), _width(), _height()], ['One geometric guided input; vertical radiation is not represented as a guided port. No fibre-coupling efficiency claim.']),
    _recipe('nanobeam_cavity', 'Photonic-crystal nanobeam cavity', 'laser-foundations', 'Two hole arrays surround an unperforated defect region.', [_p('period_um','Hole centre spacing',.6,.15,2.,.02), _p('mirror_periods','Holes per mirror',8,2,12,1,''), _p('hole_radius_um','Hole radius',.18,.02,.7,.01), _p('defect_um','Central defect length',1.,.1,8.,.1), _width(1.,'Nanobeam width'), _height()], ['Through holes are actual polygon holes with 20 vertices each. A passive cavity, not a simulated laser.']),
    _recipe('dbr_cavity', 'Passive DBR cavity', 'laser-foundations', 'Two corrugated reflector sections flank a plain guide spacer.', [_p('period_um','Mirror period',.4,.1,2.,.02), _p('mirror_periods','Periods per reflector',12,2,48,1,''), _p('duty','Duty cycle',.5,.1,.9,.05,''), _p('corrugation_um','Side corrugation',.15,.01,1.,.02), _p('spacer_um','Plain cavity spacer',2.,.1,12.,.1), _width(), _height()], ['Passive geometry only: no pumping, gain, threshold or light–current prediction.']),
    _recipe('serpentine', 'Serpentine delay layout', 'routing', 'Parallel straight segments are joined by alternating semicircular turns.', [_length(8.,1.,40.,'Straight segment length'), _p('rows','Number of straight rows',4,2,8,1,''), _p('pitch_um','Row centre spacing',2.,.2,10.,.2), _width(), _height()], ['A geometric delay path; delay, dispersion and bend loss require a qualified simulation.']),
]
_BY_ID = {item['id']:item for item in _RECIPES}


def get_catalog():
    """Return independent metadata; clients cannot mutate global recipes."""
    return deepcopy({'components':_RECIPES, 'materials':MATERIALS})


def get_materials():
    return deepcopy(MATERIALS)


def _number(value, low, high, label):
    if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be a finite number from {low} to {high}.')
    return float(value)


def _parameters(recipe, supplied):
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict) or set(supplied)-recipe['defaults'].keys():
        raise ValueError('Unknown component parameter; choose only parameters declared by this recipe.')
    result = dict(recipe['defaults'])
    for item in recipe['parameters']:
        value = _number(supplied.get(item['key'],item['default']),item['min'],item['max'],item['label'])
        if item['key'] in ('periods','mirror_periods','rows'):
            if not value.is_integer():
                raise ValueError(f"{item['label']} must be an integer.")
            value = int(value)
        result[item['key']] = value
    return result


def _point(x,y):
    return {'x':float(x),'y':float(y)}


def _arc(cx,cy,r,start,end,steps):
    return [(cx+r*math.cos(start+(end-start)*i/steps),cy+r*math.sin(start+(end-start)*i/steps)) for i in range(steps+1)]


def _path_outline(points, tangents, width):
    """Offset a sampled centreline with explicitly supplied tangent angles."""
    if len(points) != len(tangents):
        raise ValueError('Path tangent sampling does not match its coordinates.')
    left = [(x-math.sin(t)*width/2,y+math.cos(t)*width/2) for (x,y),t in zip(points,tangents)]
    right = [(x+math.sin(t)*width/2,y-math.cos(t)*width/2) for (x,y),t in zip(points,tangents)]
    return left+list(reversed(right))


def _cosine_path(x0,x1,y0,y1,width,steps=48):
    length,offset=x1-x0,y1-y0
    if length <= 0 or width*abs(offset)*math.pi**2 >= 3.2*length**2:
        raise ValueError('This bend is too tight for its width; increase transition length or reduce offset/width.')
    points=[];tangents=[]
    for i in range(steps+1):
        u=i/steps
        points.append((x0+length*u,y0+offset*(1-math.cos(math.pi*u))/2))
        tangents.append(math.atan2(offset*math.pi*math.sin(math.pi*u)/2,length))
    return _path_outline(points,tangents,width)


class _Builder:
    def __init__(self, family, parameters, material):
        self.family,self.p,self.material=family,parameters,material
        self.id='component-'+family
        self.objects=[];self.ports=[]

    def solid(self, name, kind='box', x=0.,y=0.,z=0.,sx=1.,sy=.5,sz=None, radius=1.,inner=.6,rotation=0.,vertices=None,holes=None):
        item=dict(id=f'{self.id}-{len(self.objects)+1}',name=name,visible=True,kind=kind,
                  position=dict(x=x,y=y,z=z),size=dict(x=sx,y=sy,z=self.p['height_um'] if sz is None else sz),
                  radius=radius,inner_radius=inner,rotation_deg=rotation,refractive_index=self.material['n'],
                  material_id=self.material['id'],component_id=self.id)
        if kind=='polygon':
            item['vertices']=[_point(*p) for p in vertices]
            item['holes']=[[_point(*p) for p in hole] for hole in (holes or [])]
        self.objects.append(item)
        return item

    def polygon(self,name,vertices,holes=None,**kwargs):
        return self.solid(name,kind='polygon',vertices=vertices,holes=holes,**kwargs)

    def port(self,name,x,y,angle,width=None,z=0.):
        self.ports.append(dict(name=name,x=x,y=y,z=z,angle_deg=angle,width_um=width or self.p.get('width_um',.5)))

    def two_ports(self,length,width=None):
        self.port('in',-length/2,0.,180.,width)
        self.port('out',length/2,0.,0.,width)


def _taper(x0,x1,y,w0,w1):
    return [(x0,y-w0/2),(x1,y-w1/2),(x1,y+w1/2),(x0,y+w0/2)]


def _generate(b):
    p=b.p;f=b.family;w=p.get('width_um',.5);h=p['height_um'];length=p.get('length_um',8.)
    if f=='strip':
        b.solid('Strip core',sx=length,sy=w);b.two_ports(length)
    elif f=='slot':
        gap=p['gap_um']
        for sign in (-1,1): b.solid('Slot rail',y=sign*(gap+w)/2,sx=length,sy=w)
        b.two_ports(length,2*w+gap)
    elif f=='rib':
        sh=p['slab_height_um'];sw=p['slab_width_um']
        if sh>=h or sw<=w: raise ValueError('Rib slab must be thinner than the total height and wider than the ridge.')
        b.solid('Residual slab',z=-(h-sh)/2,sx=length,sy=sw,sz=sh)
        b.solid('Raised ridge',z=sh/2,sx=length,sy=w,sz=h-sh);b.two_ports(length)
    elif f=='taper':
        a,c=p['input_width_um'],p['output_width_um']
        b.polygon('Linear width transition',_taper(-length/2,length/2,0.,a,c))
        b.port('in',-length/2,0.,180.,a);b.port('out',length/2,0.,0.,c)
    elif f=='bend':
        r=p['radius_um'];a=math.radians(p['angle_deg'])
        if r<=w/2+.02: raise ValueError('Bend radius must exceed half the width with positive inner clearance.')
        outer=_arc(0.,r,r+w/2,-math.pi/2,-math.pi/2+a,48)
        inner=_arc(0.,r,r-w/2,-math.pi/2+a,-math.pi/2,48)
        b.polygon('Circular guide bend',outer+inner)
        b.port('in',0.,0.,180.);b.port('out',r*math.sin(a),r*(1-math.cos(a)),p['angle_deg'])
    elif f=='s_bend':
        offset=p['offset_um']
        b.polygon('Cosine S-bend',_cosine_path(-length/2,length/2,-offset/2,offset/2,w,64))
        b.port('in',-length/2,-offset/2,180.);b.port('out',length/2,offset/2,0.)
    elif f=='directional_coupler':
        for sign,name in ((-1,'lower'),(1,'upper')):
            y=sign*(p['gap_um']+w)/2
            b.solid(f'{name.title()} coupling guide',y=y,sx=length,sy=w)
            b.port(name+'_in',-length/2,y,180.);b.port(name+'_out',length/2,y,0.)
    elif f=='y_branch':
        pitch=p['pitch_um']
        if pitch<=w+.05: raise ValueError('Output centre spacing must leave positive separation between branches.')
        split=-length/4
        b.solid('Input guide',x=-3*length/8,sx=length/4,sy=w)
        for sign,name in ((-1,'lower'),(1,'upper')):
            b.polygon(name.title()+' branch',_cosine_path(split,length/2,0.,sign*pitch/2,w))
            b.port(name+'_out',length/2,sign*pitch/2,0.)
        b.port('in',-length/2,0.,180.)
    elif f in ('mmi_1x2','mmi_2x2'):
        body=p['body_width_um'];pitch=p['pitch_um'];access=p['access_um'];join=1.5*w
        if pitch<=join+.05 or body<pitch+join+.05: raise ValueError('MMI body and port pitch must contain separated access tapers.')
        b.solid('Multimode body',sx=length,sy=body)
        for y in ([0.] if f=='mmi_1x2' else [-pitch/2,pitch/2]):
            b.polygon('Input access taper',_taper(-length/2-access,-length/2,y,w,join))
            b.port('in' if f=='mmi_1x2' else ('lower_in' if y<0 else 'upper_in'),-length/2-access,y,180.)
        for y in (-pitch/2,pitch/2):
            b.polygon('Output access taper',_taper(length/2,length/2+access,y,join,w))
            b.port('lower_out' if y<0 else 'upper_out',length/2+access,y,0.)
    elif f=='crossing':
        centre=p['centre_width_um']
        if centre<w or centre>=length/2: raise ValueError('Crossing central width must be at least the access width and below half its length.')
        points=[(-length/2,-w/2),(-centre,-centre/2),(centre,-centre/2),(length/2,-w/2),(length/2,w/2),(centre,centre/2),(-centre,centre/2),(-length/2,w/2)]
        b.polygon('Horizontal crossing arm',points)
        b.polygon('Vertical crossing arm',points,rotation=90.)
        b.two_ports(length);b.port('south',0.,-length/2,270.);b.port('north',0.,length/2,90.)
    elif f in ('ring_bus','ring_add_drop'):
        r=p['radius_um'];gap=p['gap_um']
        if r<=w/2+.02 or length<2*r+w+.5: raise ValueError('Ring needs positive inner radius and a bus longer than its diameter.')
        b.solid('Ring core',kind='ring',radius=r+w/2,inner=r-w/2)
        for sign,name in ([(-1,'bus')] if f=='ring_bus' else [(-1,'through'),(1,'drop')]):
            y=sign*(r+w+gap);b.solid(name.title()+' guide',y=y,sx=length,sy=w)
            b.port(name+'_left',-length/2,y,180.);b.port(name+'_right',length/2,y,0.)
    elif f=='racetrack':
        r=p['radius_um'];straight=p['straight_um']
        if r<=w/2+.02 or length<straight+2*r+w+.5: raise ValueError('Racetrack needs positive inner radius and a sufficiently long bus.')
        def capsule(radius):
            return _arc(straight/2,0.,radius,-math.pi/2,math.pi/2,32)+_arc(-straight/2,0.,radius,math.pi/2,3*math.pi/2,32)
        b.polygon('Racetrack core',capsule(r+w/2),[capsule(r-w/2)])
        y=-r-w-p['gap_um'];b.solid('Access bus',y=y,sx=length,sy=w)
        b.port('in',-length/2,y,180.);b.port('out',length/2,y,0.)
    elif f=='mzi':
        sep=p['separation_um'];transition=p['transition_um'];lead=1.;branch_length=length-2*lead
        if branch_length<=2*transition+.2 or sep<=w+.05: raise ValueError('MZI requires two transitions, a positive straight arm and separated branches.')
        if w*sep*math.pi**2>=6.4*transition**2: raise ValueError('MZI transition is too tight for the selected width and arm separation.')
        x0=-branch_length/2
        for sign,name in ((-1,'lower'),(1,'upper')):
            pts=[];tans=[];offset=sign*sep/2
            for i in range(97):
                s=branch_length*i/96
                if s<transition: u=s/transition;y=offset*(1-math.cos(math.pi*u))/2;dy=offset*math.pi*math.sin(math.pi*u)/(2*transition)
                elif s>branch_length-transition: u=(branch_length-s)/transition;y=offset*(1-math.cos(math.pi*u))/2;dy=-offset*math.pi*math.sin(math.pi*u)/(2*transition)
                else: y=offset;dy=0.
                pts.append((x0+s,y));tans.append(math.atan(dy))
            b.polygon(name.title()+' interferometer arm',_path_outline(pts,tans,w))
        b.solid('Input lead',x=-length/2+lead/2,sx=lead,sy=w)
        b.solid('Output lead',x=length/2-lead/2,sx=lead,sy=w);b.two_ports(length)
    elif f in ('bragg','dbr_cavity'):
        period=p['period_um'];duty=p['duty'];corr=p['corrugation_um'];lead=1.
        if f=='bragg':
            n=p['periods'];span=n*period;centres=[-span/2+(i+.5)*period for i in range(n)];total=span+2*lead
        else:
            n=p['mirror_periods'];spacer=p['spacer_um'];total=2*n*period+spacer+2*lead
            centres=[sign*(spacer/2+(i+.5)*period) for sign in (-1,1) for i in range(n)]
        b.solid('Continuous guide',sx=total,sy=w)
        for x in centres: b.solid('Width corrugation',x=x,sx=period*duty,sy=w+2*corr)
        b.two_ports(total)
    elif f=='grating_coupler':
        period=p['period_um'];n=p['periods'];ap=p['aperture_um'];taper=p['taper_um'];sh=p['slab_height_um'];span=n*period;lead=1.
        if sh>=h or ap<=w: raise ValueError('Grating residual slab must be thinner than the core and aperture wider than the input.')
        b.solid('Input lead',x=-taper-lead/2,sx=lead,sy=w)
        b.polygon('Access aperture taper',_taper(-taper,0.,0.,w,ap))
        b.solid('Residual grating slab',x=span/2,z=-(h-sh)/2,sx=span,sy=ap,sz=sh)
        for i in range(n): b.solid('Grating ridge',x=(i+.5)*period,z=sh/2,sx=period*p['duty'],sy=ap,sz=h-sh)
        b.port('in',-taper-lead,0.,180.)
    elif f=='nanobeam_cavity':
        period=p['period_um'];n=p['mirror_periods'];r=p['hole_radius_um'];defect=p['defect_um'];total=2*n*period+defect+2.
        if 2*r>=min(w,period)-.02: raise ValueError('Nanobeam holes need positive wall and inter-hole clearance.')
        holes=[]
        for sign in (-1,1):
            for i in range(n):
                x=sign*(defect/2+(i+.5)*period)
                holes.append(_arc(x,0.,r,0.,2*math.pi,20)[:-1])
        b.polygon('Perforated nanobeam',_taper(-total/2,total/2,0.,w,w),holes);b.two_ports(total)
    elif f=='serpentine':
        rows=p['rows'];pitch=p['pitch_um'];r=pitch/2
        if pitch<=w+.05: raise ValueError('Serpentine row spacing must leave positive clearance between guides.')
        pts=[];tans=[]
        def add(point,angle):
            if pts and math.dist(point,pts[-1])<1e-10:
                tans[-1]=angle
            else:pts.append(point);tans.append(angle)
        for row in range(rows):
            y=(row-(rows-1)/2)*pitch;direction=1 if row%2==0 else -1
            add((-direction*length/2,y),0. if direction==1 else math.pi)
            add((direction*length/2,y),0. if direction==1 else math.pi)
            if row<rows-1:
                start=-math.pi/2;end=math.pi/2 if direction==1 else -3*math.pi/2
                for i in range(1,13):
                    theta=start+(end-start)*i/12
                    add((direction*length/2+r*math.cos(theta),y+r+r*math.sin(theta)),theta+(math.pi/2 if direction==1 else -math.pi/2))
        b.polygon('Continuous serpentine guide',_path_outline(pts,tans,w))
        b.port('in',-length/2,-(rows-1)*pitch/2,180.)
        sign=1 if (rows-1)%2==0 else -1
        b.port('out',sign*length/2,(rows-1)*pitch/2,0. if sign==1 else 180.)
    else:
        raise ValueError('Unknown component recipe.')


def _transform(x,y,angle):
    c,s=math.cos(angle),math.sin(angle)
    return c*x-s*y,s*x+c*y


def _bounds(item):
    p=item['position'];a=math.radians(item['rotation_deg'])
    if item['kind'] in ('ring','cylinder'):
        r=item['radius'];return (p['x']-r,p['x']+r,p['y']-r,p['y']+r,p['z']-item['size']['z']/2,p['z']+item['size']['z']/2)
    local=([(v['x'],v['y']) for v in item['vertices']] if item['kind']=='polygon' else
           [(x*item['size']['x']/2,y*item['size']['y']/2) for x in (-1,1) for y in (-1,1)])
    points=[_transform(x,y,a) for x,y in local]
    return (min(x for x,_ in points)+p['x'],max(x for x,_ in points)+p['x'],min(y for _,y in points)+p['y'],max(y for _,y in points)+p['y'],p['z']-item['size']['z']/2,p['z']+item['size']['z']/2)


def build_component(component_id, parameters=None, placement=None, material_id='generic-core'):
    """Build one deterministic instance and a containing design-only document."""
    if not isinstance(component_id,str) or component_id not in _BY_ID:
        raise ValueError('Unknown photonic component family.')
    recipe=_BY_ID[component_id];params=_parameters(recipe,parameters)
    if not isinstance(material_id,str): raise ValueError('Material id must select a catalog material.')
    material=next((m for m in MATERIALS if m['id']==material_id),None)
    if material is None: raise ValueError('Unknown catalog material. Assign a custom material after placing the component.')
    if placement is None: placement={}
    if not isinstance(placement,dict) or set(placement)-{'x','y','z','rotation_deg'}: raise ValueError('Placement accepts x, y, z and rotation_deg only.')
    placement={key:_number(placement.get(key,0.),-360 if key=='rotation_deg' else -450.,360 if key=='rotation_deg' else 450.,key) for key in ('x','y','z','rotation_deg')}
    placement['rotation_deg']%=360
    b=_Builder(component_id,params,material);_generate(b)
    angle=math.radians(placement['rotation_deg'])
    for obj in b.objects:
        x,y=_transform(obj['position']['x'],obj['position']['y'],angle)
        obj['position']={'x':x+placement['x'],'y':y+placement['y'],'z':obj['position']['z']+placement['z']}
        obj['rotation_deg']=(obj['rotation_deg']+placement['rotation_deg'])%360
    for port in b.ports:
        x,y=_transform(port['x'],port['y'],angle)
        port.update(x=x+placement['x'],y=y+placement['y'],z=port['z']+placement['z'],angle_deg=(port['angle_deg']+placement['rotation_deg'])%360)
    bounds=[_bounds(obj) for obj in b.objects]
    domain={axis:max(2.,2*(max(abs(v[i]) for v in bounds for i in (2*j,2*j+1))+1.)) for j,axis in enumerate('xyz')}
    component=dict(id=b.id,family=component_id,generator_version=GENERATOR_VERSION,parameters=params,
                   object_ids=[obj['id'] for obj in b.objects],placement=placement,material_id=material_id,ports=b.ports)
    geometry=dict(schema='photonics-workbench-geometry',schema_version=2,units='um',name=recipe['name'],mode='draft-3d',
                  domain=domain,objects=b.objects,materials=[deepcopy(material)],components=[component])
    from .geometry_draft import validate_geometry
    geometry=validate_geometry(geometry)
    return {'geometry':geometry,'component':deepcopy(geometry['components'][0])}
