"""Bounded polygon and physical port geometry for a planar Ez scattering study."""
import math
import re

from .geometry_draft import validate_polygon

HALF_CELL = 4.
PML = 1.
APERTURE = 2.
INWARD_EXTENSION = 1.
OUTWARD = {"-x":(0,-1), "+x":(0,1), "-y":(1,-1), "+y":(1,1)}
TOL = 1e-9


def default_planar_port_layout():
    # A square inner corner and two straight sections joining the port leads.
    w = .375
    return {"version":1,"name":"Silicon right-angle elbow",
            "polygons":[[[-1.,-w/2],[w/2,-w/2],[w/2,1.],[-w/2,1.],[-w/2,w/2],[-1.,w/2]]],
            "ports":[{"name":"west","center_um":[-1.75,0.],"outward":"-x","width_um":w},
                     {"name":"north","center_um":[0.,1.75],"outward":"+y","width_um":w}]}


def _pair(value, limit):
    if not isinstance(value,list) or len(value) != 2 or any(
        type(v) not in (int,float) or abs(v)>limit or not math.isfinite(v) for v in value):
        raise ValueError(f"Expected two finite coordinates within ±{limit} µm.")
    return list(map(float,value))


def _text(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,62}",value):
        raise ValueError("Use a plain 1–63 character name for a planar design or port.")
    return value


def lead_bounds(port, *, aperture=False):
    """Axis-aligned Si lead or empty required monitor corridor, as [xmin,xmax,ymin,ymax]."""
    axis, sign = OUTWARD[port["outward"]]
    s, t = port["center_um"][axis], port["center_um"][1-axis]
    start = s - sign*(.25 if aperture else INWARD_EXTENSION)
    normal = sorted((start,sign*HALF_CELL))
    width = APERTURE if aperture else port["width_um"]
    transverse = [t-width/2,t+width/2]
    return normal+transverse if axis==0 else transverse+normal


def _covers_section(polygons, port):
    axis, sign = OUTWARD[port["outward"]]
    s = port["center_um"][axis] - sign*INWARD_EXTENSION
    lo = port["center_um"][1-axis] - port["width_um"]/2
    hi = lo + port["width_um"]
    intervals = []
    # Half-open crossings pair into inside intervals for even concave simple contours.
    for outline in polygons:
        crossings = []
        for a, b in zip(outline, outline[1:]+outline[:1]):
            if abs(a[axis]-s)<TOL and abs(b[axis]-s)<TOL:
                intervals.append((min(a[1-axis],b[1-axis]), max(a[1-axis],b[1-axis])))
            if (a[axis] <= s < b[axis]) or (b[axis] <= s < a[axis]):
                ratio = (s-a[axis])/(b[axis]-a[axis])
                crossings.append(a[1-axis] + ratio*(b[1-axis]-a[1-axis]))
        crossings.sort()
        intervals.extend(zip(crossings[::2],crossings[1::2]))
    covered = lo
    for a,b in sorted(intervals):
        if a > covered + TOL:
            return False
        covered = max(covered,b)
        if covered >= hi-TOL:
            return True
    return False


def validate_planar_port_layout(value):
    if not isinstance(value,dict) or set(value)!={"version","name","polygons","ports"}:
        raise ValueError("A planar layout requires version, name, polygons and ports.")
    if type(value["version"]) is not int or value["version"]!=1:
        raise ValueError("Unsupported planar layout version.")
    name = _text(value["name"])
    if not isinstance(value["polygons"],list) or not 1<=len(value["polygons"])<=8:
        raise ValueError("Supply between one and eight silicon polygon outlines.")
    polygons = []
    for outline in value["polygons"]:
        if not isinstance(outline,list) or not 3<=len(outline)<=64:
            raise ValueError("Each polygon requires 3–64 vertices.")
        vertices = [_pair(point,2.5) for point in outline]
        canonical, _ = validate_polygon([{"x":x,"y":y} for x,y in vertices],[])
        polygons.append([[point["x"],point["y"]] for point in canonical])
    if not isinstance(value["ports"],list) or not 2<=len(value["ports"])<=4:
        raise ValueError("A planar layout requires two to four guided ports.")
    ports = []
    for p in value["ports"]:
        if not isinstance(p,dict) or set(p)!={"name","outward","center_um","width_um"}:
            raise ValueError("Each port requires name, outward, center_um and width_um.")
        key, direction = _text(p["name"]), p["outward"]
        if not isinstance(direction,str) or direction not in OUTWARD:
            raise ValueError("Choose a port direction from -x, +x, -y or +y.")
        if any(q["name"]==key or q["outward"]==direction for q in ports):
            raise ValueError("Each port needs a unique name and outward direction.")
        center = _pair(p["center_um"],2.25)
        width = p["width_um"]
        if type(width) not in (int,float) or not .25<=width<=.5:
            raise ValueError("Port widths range from 0.25 to 0.50 µm.")
        axis, sign = OUTWARD[direction]
        radius = sign*center[axis]
        if not 1.25<=radius<=2.25 or abs(center[1-axis])>1.5:
            raise ValueError("Each port plane must be 1.25–2.25 µm outward and within ±1.5 µm transversely.")
        if any(sign*point[axis]>radius-.25+TOL for poly in polygons for point in poly):
            raise ValueError("Leave 0.25 µm of straight lead between every polygon and each physical plane.")
        port = {"name":key,"outward":direction,"center_um":center,"width_um":float(width)}
        if not _covers_section(polygons,port):
            raise ValueError(f"Port {key} needs a full-width silicon attachment one µm inward from its plane.")
        ports.append(port)
    for i, port in enumerate(ports):
        aperture = lead_bounds(port,aperture=True)
        for other in ports[:i]+ports[i+1:]:
            lead = lead_bounds(other)
            if all(min(aperture[n+1],lead[n+1])-max(aperture[n],lead[n])>TOL for n in (0,2)):
                raise ValueError("A different port's silicon lead intersects a source or monitor corridor.")
    return {"version":1,"name":name,"polygons":polygons,"ports":ports}
