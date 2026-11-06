"""Export plane-registered complex measurements and their original phase validity masks."""
import csv
import io
import math


def registered_csv(study):
    ports={'ring':('reflection','through','drop','add_side'),'grating':('reflection','guide')}[study['component']]
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(('vendor_geometry','mesh_cells_per_um','native_arrays_sha256','basis','wavelength_um','port',
                     'real','imag','power_fraction','native_spectrum_valid','phase_reportable','phase_rad',
                     'signed_phase_minus_vendor_rad'))
    for s in study['native_samples']:
        sources=[('frequency-specific',s['registered_complex_s_real'],s['registered_complex_s_imag'],s['valid_bins'],s['phase_ok'],None)]
        central=s.get('single_central_basis_band')
        if central:
            sources.append(('single central mode',central['registered_native_real'],central['registered_native_imag'],
                            central['native_valid_bins'],central['native_phase_ok'],central['signed_native_minus_vendor_phase_rad']))
        for basis,reals,imags,valid_bins,phases,residuals in sources:
            for i,(wave,valid,real,imag,usable) in enumerate(zip(s['wavelength_um'],valid_bins,reals,imags,phases,strict=True)):
                for j,(port,re,im,allowed) in enumerate(zip(ports,real,imag,usable,strict=True)):
                    allowed=bool(valid and allowed and math.hypot(re,im)>=study['amplitude_floor'])
                    residual=residuals[i][j] if allowed and residuals is not None else None
                    writer.writerow((study['vendor'],s['resolution'],s['arrays_sha256'],basis,wave,port,
                                     re,im,re*re+im*im,bool(valid),allowed,math.atan2(im,re) if allowed else None,residual))
    return stream.getvalue()
