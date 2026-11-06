"""Independent scalar frequency-domain solve for planar Ez polygon port benchmarks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import planar_port_layout as layout


def permittivity(design, axis, subpixels, launch=None):
    from matplotlib.path import Path as Polygon

    h=axis[1]-axis[0]
    xx,yy=np.meshgrid(axis,axis,indexing="ij")
    mask=np.zeros(xx.shape,dtype=float)
    polygons=[Polygon(np.asarray(vertices),closed=False) for vertices in design["polygons"]]
    leads=[layout.lead_bounds(p) for p in design["ports"]]
    for offset_x in (np.arange(subpixels)+.5)/subpixels-.5:
        for offset_y in (np.arange(subpixels)+.5)/subpixels-.5:
            x,y=xx+offset_x*h,yy+offset_y*h
            if launch is None:
                points=np.column_stack((x.ravel(),y.ravel()))
                silicon=np.zeros(x.size,dtype=bool)
                for poly in polygons:
                    silicon|=poly.contains_points(points)
                silicon=silicon.reshape(x.shape)
                for x0,x1,y0,y1 in leads:
                    silicon|=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
            else:
                normal,_=layout.OUTWARD[launch["outward"]]
                coord=y if normal==0 else x
                silicon=abs(coord-launch["center_um"][1-normal])<launch["width_um"]/2
            mask+=silicon/subpixels**2
    return 1.44**2+(3.48**2-1.44**2)*mask


def helmholtz_factor(eps, axis, k):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    n=len(axis)
    h=axis[1]-axis[0]
    def stretch(grid):
        return 1+6j*np.maximum(abs(grid)-(layout.HALF_CELL-layout.PML),0)**3
    stretch_center=stretch(axis)
    stretch_edge=stretch(axis+h/2)
    xplus=stretch_center[None,:]/stretch_edge[:,None]/h**2
    xminus=stretch_center[None,:]/stretch(axis-h/2)[:,None]/h**2
    yplus=xplus.T
    yminus=xminus.T
    diag=k*k*eps*stretch_center[:,None]*stretch_center[None,:]-xplus-xminus-yplus-yminus
    ind=np.arange(n*n).reshape(n,n)
    rows=np.concatenate([ind.ravel(),ind[:-1].ravel(),ind[1:].ravel(),ind[:,:-1].ravel(),ind[:,1:].ravel()])
    cols=np.concatenate([ind.ravel(),ind[1:].ravel(),ind[:-1].ravel(),ind[:,1:].ravel(),ind[:,:-1].ravel()])
    vals=np.concatenate([diag.ravel(),xplus[:-1].ravel(),xminus[1:].ravel(),yplus[:,:-1].ravel(),yminus[:,1:].ravel()])
    return splu(coo_matrix((vals,(rows,cols)),shape=(n*n,n*n)).tocsc(),permc_spec="MMD_AT_PLUS_A")


def basis(port, axis, k):
    from scipy.linalg import eigh_tridiagonal

    normal,_=layout.OUTWARD[port["outward"]]
    h=axis[1]-axis[0]
    t=port["center_um"][1-normal]
    lengths=np.maximum(0,np.minimum(axis+h/2,t+port["width_um"]/2)-np.maximum(axis-h/2,t-port["width_um"]/2))
    eps=1.44**2+(3.48**2-1.44**2)*lengths/h
    vals,vecs=eigh_tridiagonal(k*k*eps-2/h**2,np.ones(len(axis)-1)/h**2,
                               select="i",select_range=(len(axis)-1,len(axis)-1))
    beta=2*np.arcsin(np.sqrt(vals[0])*h/2)/h
    phi=vecs[:,0]*np.sign(vecs[:,0].sum())
    phase_gradient=np.sin(beta*h)/h
    phi/=np.sqrt(phase_gradient*np.sum(phi*phi)*h)
    return phi,phase_gradient


def project(field, port, axis, phi, gradient):
    normal,_=layout.OUTWARD[port["outward"]]
    h=axis[1]-axis[0]
    index=int(np.argmin(abs(axis-port["center_um"][normal])))
    if abs(axis[index]-port["center_um"][normal])>1e-7:
        raise ValueError("Independent solver requires a grid-aligned reference plane.")
    slices=field if normal==0 else field.T
    e=slices[index]
    slope=(slices[index+1]-slices[index-1])/(2*h)
    return np.array([np.dot(phi,e+slope/(1j*gradient)),np.dot(phi,e-slope/(1j*gradient))])*gradient*h/2


def measure(design, mesh, waves):
    ports=design["ports"]
    # Integral grid coordinates allow modal amplitudes at the exact declared planes.
    axis=np.arange(-layout.HALF_CELL*mesh+1,layout.HALF_CELL*mesh)/mesh
    eps=permittivity(design,axis,8)
    straight=[permittivity(design,axis,8,p) for p in ports]
    matrix=np.empty((len(waves),len(ports),len(ports)),complex)
    for f,w in enumerate(waves):
        print(f"Independent frequency solve at {w:.3f} µm, {mesh} cells/µm",flush=True)
        k=2*np.pi/w
        modes=[basis(p,axis,k) for p in ports]
        device=helmholtz_factor(eps,axis,k)
        for i,p in enumerate(ports):
            normal,sign=layout.OUTWARD[p["outward"]]
            at=p["center_um"][normal]+sign*.5
            line=int(np.argmin(abs(axis-at)))
            current=np.zeros(eps.shape,complex)
            if normal==0:
                current[line,:]=modes[i][0]
            else:
                current[:,line]=modes[i][0]
            field=device.solve(current.ravel()).reshape(eps.shape)
            straight_factor=helmholtz_factor(straight[i],axis,k)
            ref=straight_factor.solve(current.ravel()).reshape(eps.shape)
            del straight_factor
            reference=project(ref,p,axis,*modes[i])
            incident=reference[int(sign>0)]
            for j,q in enumerate(ports):
                branch=int(layout.OUTWARD[q["outward"]][1]<0)
                out=project(field,q,axis,*modes[j])[branch]
                if i==j:
                    out-=reference[branch]
                matrix[f,j,i]=out/incident
    return matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--resolution",type=int,choices=(24,32,40),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path.")
    design=layout.validate_planar_port_layout(json.loads(args.design.read_text()))
    waves=np.array([1.60,1.55,1.50])
    matrix=measure(design,args.resolution,waves)
    args.output.write_text(json.dumps({"geometry":design,"solver":"independent 2D scalar Ez frequency-domain Helmholtz",
        "cells_per_um":args.resolution,"wavelength_um":waves.tolist(),"s_real":matrix.real.tolist(),
        "s_imag":matrix.imag.tolist(),"power":(abs(matrix)**2).tolist()},indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
