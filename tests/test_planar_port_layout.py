import copy
import unittest

from backend.planar_port_layout import default_planar_port_layout, validate_planar_port_layout


class PlanarPortLayoutTests(unittest.TestCase):
    def test_default_and_reversed_winding_are_canonical(self):
        value=default_planar_port_layout()
        reverse=copy.deepcopy(value)
        reverse["polygons"][0].reverse()
        self.assertEqual(validate_planar_port_layout(value),validate_planar_port_layout(reverse))

    def test_ports_can_move_on_different_axes_and_touch_a_polygon_edge(self):
        value=default_planar_port_layout()
        value["ports"][0]["center_um"][0]=-2.
        value["ports"][1]["center_um"][1]=1.5
        result=validate_planar_port_layout(value)
        self.assertEqual([p["center_um"] for p in result["ports"]],[[-2.,0.],[0.,1.5]])

    def test_crossed_outline_unknown_direction_and_disconnected_lead(self):
        cases=[]
        bow=default_planar_port_layout()
        bow["polygons"]=[[[-1.,-1.],[1.,1.],[-1.,1.],[1.,-1.]]]
        cases.append(bow)
        for field,val in [("outward","diagonal"),("width_um",.9),("center_um",[-1.75,.3])]:
            invalid=default_planar_port_layout()
            invalid["ports"][0][field]=val
            cases.append(invalid)
        invalid=default_planar_port_layout()
        invalid["ports"][1]["name"]=invalid["ports"][0]["name"]
        cases.append(invalid)
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_planar_port_layout(value)

    def test_polygon_intruding_on_the_monitor_corridor_is_rejected(self):
        value=default_planar_port_layout()
        value["polygons"].append([[0.,1.4],[.4,1.7],[-.1,1.6]])
        with self.assertRaisesRegex(ValueError,"straight lead"):
            validate_planar_port_layout(value)

    def test_a_lead_cannot_join_only_at_one_point(self):
        value=default_planar_port_layout()
        value["polygons"][0][0][1]=0
        with self.assertRaisesRegex(ValueError,"full-width"):
            validate_planar_port_layout(value)
