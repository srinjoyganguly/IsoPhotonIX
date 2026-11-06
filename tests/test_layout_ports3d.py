"""Regression checks for signed cardinal-port frames and exact straight corridors."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend.planar_port_layout import default_planar_port_layout, validate_planar_port_layout
from scripts.calibrate_layout_ports3d import frame, gpu_lock_path, grid, local_phasors, modal_amplitudes, permittivity


class LayoutPorts3DTests(unittest.TestCase):
    def test_lock_override_and_private_per_user_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ,{'PHOTONICS_GPU_LOCK':str(root/'shared.lock'),'XDG_RUNTIME_DIR':tmp}):
                self.assertEqual(gpu_lock_path(),root/'shared.lock')
            with patch.dict(os.environ,{'PHOTONICS_GPU_LOCK':'','XDG_RUNTIME_DIR':tmp}):
                self.assertEqual(gpu_lock_path(),root/'photonics-workbench-gpu.lock')
            with patch.dict(os.environ,{'PHOTONICS_GPU_LOCK':'','XDG_RUNTIME_DIR':''}), patch('tempfile.gettempdir',return_value=tmp):
                lock = gpu_lock_path()
                self.assertEqual(lock.parent,root/f'photonics-workbench-{os.getuid()}')
                self.assertEqual(lock.parent.stat().st_mode & 0o077,0)

    def test_y_port_uses_positive_ex_and_axial_h_sign(self):
        # A +y wave with +Ex has -Hz. Its backward partner keeps Ex and reverses H.
        electric = np.zeros((3,5,1,4))
        magnetic = np.zeros_like(electric)
        electric[0] = 1.
        magnetic[2] = -1.
        local_e = frame(electric,1)
        local_h = frame(magnetic,1,axial=True)
        np.testing.assert_array_equal(local_e[1],1)
        np.testing.assert_array_equal(local_h[2],1)
        np.testing.assert_equal(frame(local_h,1,axial=True),magnetic)
        for sign in (1,-1):
            global_fields = np.concatenate((electric,sign*magnetic))[None]
            local = local_phasors(global_fields,1)
            amplitudes,power = modal_amplitudes(local,local_e[:,0][None],local_h[:,0][None],1.)
            expected = np.zeros(2)
            expected[int(sign<0)] = np.sqrt(power[0])
            np.testing.assert_allclose(amplitudes[0],expected)

    def test_four_displaced_ports_keep_distinct_straight_monitor_sections(self):
        ports = [{'name':name,'outward':direction,'center_um':center,'width_um':width}
                 for name,direction,center,width in (
                     ('west','-x',[-1.79,-.23],.375),('east','+x',[1.84,.31],.43),
                     ('south','-y',[-.29,-1.88],.32),('north','+y',[.14,1.82],.50))]
        design = validate_planar_port_layout({'version':1,'name':'Four displaced leads','ports':ports,
            'polygons':[[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]]})
        mesh = 12
        _,physical = grid(design,mesh)
        body = permittivity(design,mesh,0,False)
        for i,p in enumerate(physical):
            reference = permittivity(design,mesh,i,True)
            self.assertLessEqual(abs(p['sampled_center_um'][p['axis']]-p['center_um'][p['axis']]),.5/mesh)
            for name in ('source','start'):
                crop = (slice(None),*tuple(slice(s,s+n) for s,n in zip(p[name],p['shape'])))
                np.testing.assert_array_equal(body[crop],reference[crop])

    def test_extrusion_and_attachment_match_at_all_source_and_monitor_slices(self):
        design = default_planar_port_layout()
        mesh = 12
        _,ports = grid(design,mesh)
        body = permittivity(design,mesh,0,False)
        for launch,p in enumerate(ports):
            reference = permittivity(design,mesh,launch,True)
            for name in ('source','start'):
                cuts = tuple(slice(s,s+n) for s,n in zip(p[name],p['shape']))
                section = (slice(None),*cuts)
                np.testing.assert_array_equal(body[section],reference[section])
        # The mirrored straight rails have exactly equivalent permittivity tensors.
        x = permittivity(design,mesh,0,True)
        y = permittivity(design,mesh,1,True)
        np.testing.assert_array_equal(frame(y,1),x)


if __name__=='__main__':
    unittest.main()
