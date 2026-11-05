"""Build the local GUI's reference tables from the committed numerical receipts."""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'docs/verification'


def read(name):
    path=BASE/name
    return json.loads(path.read_text()) if path.is_file() else None


def matrix_view(report):
    return {key:report[key] for key in ('ports','port_order','channels','wavelength_um','s','s_real','s_imag','power')
            if key in report}


def add_matrix_case(key, references, summary, dimension, limits, comparisons):
    records=[]
    for label,filename in references:
        report=read(filename)
        if report is None:continue
        if not all(report.get(key,True) for key in ('passed','all_stages_settled','settled','full_matrix')):
            continue
        records.append({'label':label,'receipt':filename,'report':matrix_view(report)})
    if not records:return None
    return {'summary':summary,'dimension':dimension,'routes':[r['label'] for r in records],
            'limits':limits,'comparisons':comparisons,'recorded':records,'series':[],'wavelengths_um':[]}


def vendor(vendor,url,description):
    return dict(vendor=vendor,url=url,description=description)


def build():
    index={}
    for key in ('coupler','splitter'):
        refs=[('Meep CPU',f'v011-meep-multiport-{key}-electric.json')]
        for device,label in [('gpu','CUDA'),('cpu','CPU')]:
            for mesh in (16,12):
                refs.append((f'FDTDX {label} · aligned grid · {mesh} cells/µm',f'v011-fdtdx-yee-{key}-{device}{mesh}.json'))
            refs.append((f'FDTDX {label} · original grid',f'v011-fdtdx-{key}-{device}.json'))
        if key=='coupler':
            summary='Each input illuminates a separate access guide. Select an input and a wavelength to inspect every output, including reflection.'
            comparisons=[vendor('Lumerical','https://optics.ansys.com/hc/en-us/articles/360042304694-Evanescent-waveguide-couplers',
                      'The published SOI coupling length is 12.8827 µm for a 50 nm gap. Our measured four-port circuit uses a 125 nm central gap and curved access guides.'),
                        vendor('Tidy3D','https://www.flexcompute.com/tidy3d/examples/notebooks/DirectionalCoupler/',
                      'Compare straight-section supermodes separately from the complete bent circuit. Matched local Tidy3D mode measurements are recorded in the vendor comparison guide.')]
        else:
            summary='The stem and both output arms were driven independently. Total collected guided power includes exactly the monitored channels.'
            comparisons=[vendor('Lumerical','https://optics.ansys.com/hc/en-us/articles/360042800353-Y-branch',
                              'The published optimized Y splits the transmitted power approximately in half. Compare arm imbalance separately from insertion loss.'),
                         vendor('Tidy3D','https://www.flexcompute.com/tidy3d/examples/notebooks/YJunction/',
                              'The published thirteen-width junction reports <0.2 dB mean excess loss over 1.50–1.60 µm. Our cosine branch has a different shape.')]
        entry=add_matrix_case(key,refs,summary,'Finite-height 3D',
                'Fundamental quasi-TE mode at each port. Radiation and higher modes are excluded from the reported guided power sum. Material-grid differences affect weak reflection, phase and arm symmetry.',comparisons)
        if entry:index[key]=entry
    for key,file in [('multimode','v011-multimode-planar.json'),('multimode-3d','v011-multimode-finite.json')]:
        entry=add_matrix_case(key,[('Meep CPU',file)],'Two modes enter independently from each side of an offset joint. The reference planes include the straight access lengths.',
            'Planar invariant-z slab' if key=='multimode' else 'Finite-height silicon strip',
            'The matrix includes two co-polarized guided modes per side. Radiation is unobserved. The planar study also has an independent transverse open-boundary mode-matching solution.',
            [vendor('Lumerical','https://optics.ansys.com/hc/en-us/articles/360034396614-MODE-EigenMode-Expansion-EME-solver-introduction',
                    'The vendor EME method matches tangential fields between sections. Published slab-mode values are separate checks; there is no matched Lumerical matrix for this offset.'),
             vendor('Tidy3D','https://www.flexcompute.com/tidy3d/examples/notebooks/EMESolver/',
                    'Local Tidy3D port modes use our exact cross sections. The published EME tutorial demonstrates modal coupling in a different device.')])
        if entry:index[key]=entry
    for key,file in [('ring','v011-ring.json'),('grating','v011-grating.json')]:
        report=read(file)
        if key=='ring' and (report is None or not report.get('passed')):
            stored=[]
            comparisons=[]
            for name,brand in [('lumerical','Lumerical'),('tidy3d','Tidy3D')]:
                filename=f'v011-ring-loaded-{name}-fine.json'
                native=read(filename)
                if native is None or not native.get('passed'):
                    url=('https://www.flexcompute.com/tidy3d/examples/notebooks/RingResonator/' if name=='tidy3d'
                         else 'https://optics.ansys.com/hc/en-us/articles/360042800213-Ring-resonator-getting-started-Final-parameter-extraction')
                    comparisons.append(vendor(brand,url,'The published loaded-ring spectrum is the reference; local settling and mesh checks are in progress.'))
                    continue
                row=native['series'][name]
                view={'series':{name:{k:row[k] for k in ('wavelength_um','through','drop','reflection','add_side','valid_bins')}}}
                stored.append({'label':f'{brand} geometry · Meep CPU','receipt':filename,'report':view})
                comparisons.append(vendor(brand,native['physical_model']['source'],
                    f"Measured planar mode spacing {native['measured_loaded_fsr_nm']:.2f} nm. The published loaded-ring spacing is {native['vendor_loaded_fsr_nm']:.2f} nm."))
            if stored:
                index[key]={'dimension':'Planar effective-index add/drop model','routes':['Meep CPU'],'series':[],'wavelengths_um':[],
                            'summary':'Both buses are monitored over the spectral band.',
                            'comparisons':comparisons,'limits':'A finite-height isolated-ring pole and a resonance of a loaded ring have different boundary conditions.',
                            'recorded':stored}
            continue
        if key=='grating' and (report is None or not report.get('passed')):
            compact=read('v011-grating-compact-tidy3d.json')
            if compact and compact.get('passed'):
                view={'compact_tidy_domain':{n:compact['result'][n] for n in ('wavelength_um','beam_to_guide','valid_beam_bins') if n in compact['result']}}
                index[key]={'dimension':'Planar grating with finite collection aperture','routes':['Meep CPU'],
                            'series':[],'wavelengths_um':[],'summary':'Downward Gaussian illumination is calibrated to the incident beam power.',
                            'limits':'The solved domain, finite source aperture and outgoing guided mode define this efficiency.',
                            'comparisons':[vendor('Tidy3D',compact['source_url'],'Measured 58.23% against 56.23% at 1.55 µm for this compact setup.'),
                                           vendor('Lumerical','https://optics.ansys.com/hc/en-us/articles/360042304774-Broadband-grating-coupler-2D',
                                                  'The 1.31 µm Lumerical grating has a 30 nm etched groove. At 40 cells/µm Meep measured 49.42% at the nominal wavelength; Lumerical reports 55.7%.')],
                            'recorded':[{'label':'Meep CPU','receipt':'v011-grating-compact-tidy3d.json','report':view}]}
            continue
        if report is None or not report.get('passed'):continue
        comparisons=[]
        for row in report['comparisons']:
            if key=='ring':
                desc=f"Measured planar free spectral range {row['local_fsr_nm']:.3f} nm; related vendor example {row['reference_fsr_nm']:.3f} nm."
            else:
                desc=f"Measured Gaussian-to-guide power {100*row['measured_fraction']:.2f}%; vendor example {100*row['reference_fraction']:.2f}%. {row['scope']}"
            comparisons.append(vendor(row['vendor'],row['source_url'],desc))
        if key=='ring':
            view={'series':{k:{n:r[n] for n in ('wavelength_um','through','reflection','drop','add_side','valid_bins')} for k,r in report['series'].items()}}
            view['isolated_finite_height_ring']={k:{n:r[n] for n in ('measured_bare_ring_fsr_nm','published_loaded_ring_fsr_nm')} for k,r in report.get('isolated_finite_height_ring',{}).items()}
        else:
            view={'studies':{k:{'fine':{n:r['fine'][n] for n in ('wavelength_um','beam_to_guide','guide_radiation_up','guide_radiation_down','valid_beam_bins','valid_energy_bins') if n in r['fine']}} for k,r in report['studies'].items()}}
            compact=report.get('compact_tidy_domain')
            if compact:
                view['compact_tidy_domain']={n:compact[n] for n in ('wavelength_um','beam_to_guide','valid_beam_bins') if n in compact}
                comparisons.append(vendor('Tidy3D compact aperture','https://www.flexcompute.com/tidy3d/examples/notebooks/GratingCoupler/',
                    f"Matching the compact collection domain gives {compact['center_beam_to_guide']*100:.2f}% at 1.55 µm. Tidy3D reports 56.23%."))
        index[key]={'dimension':report.get('dimensionality','Planar effective-index add/drop model'),
                   'routes':['Meep CPU'],'series':[],'wavelengths_um':[],'comparisons':comparisons,
                   'summary':'Inspect independently calibrated spectra for both published reference geometries.',
                   'limits':('A finite-height isolated ring check excludes the bus waveguides. A full loaded 3D spectrum requires a separate calculation.' if key=='ring'
                            else 'Upward radiated power and power coupled to a particular Gaussian mode have different collection and normalization.'),
                   'recorded':[{'label':'Meep CPU','receipt':file,'report':view}]}
    if 'grating' in index:
        for label,name in [('Lumerical article · 28 cells/µm','v011-grating-ansys-article-r28.json'),
                           ('Lumerical article · 40 cells/µm','v011-grating-ansys-article-r40.json')]:
            beam=read(name)
            if beam and beam.get('passed'):
                panel={'case_id':beam['case_id'],'vendor':beam['vendor'],
                       'result':{k:beam['result'][k] for k in ('wavelength_um','beam_to_guide','valid_beam_bins')}}
                index['grating']['recorded'].append({'label':label,'receipt':name,'report':panel})
    return index


if __name__=='__main__':
    (BASE/'v011-calibrated-index.json').write_text(json.dumps(build(),indent=2,allow_nan=False)+'\n')
