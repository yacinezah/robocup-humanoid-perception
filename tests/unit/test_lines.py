import unittest,json
from pathlib import Path
import numpy as np
from robocup_perception.localization.lines import support_pixels,project_support,line_potential

class LineTests(unittest.TestCase):
    def setUp(self):
        self.lines=np.array([[[-3.,0],[3.,0]],[[0,-3.],[0,3.]]])
        self.f={'robot_xy_m':[[.2,0],[.4,0],[.6,0],[.8,0]],'confidence':[.9]*4}
    def test_alignment(self):
        score=line_potential([[0,0,0],[0,1.5,0]],self.f,self.lines,.75)
        self.assertGreater(score[0],score[1]);self.assertTrue(np.isfinite(score).all())
    def test_empty_invalid_neutral(self):
        for f in [{'robot_xy_m':[],'confidence':[]},{'robot_xy_m':[[float('nan'),0]]*4,'confidence':[.8]*4}]:
            np.testing.assert_array_equal(line_potential([[0,0,0]],f,self.lines,.75),[0.])
    def test_duplicate_cloud(self):
        f={k:v*8 for k,v in self.f.items()}
        np.testing.assert_allclose(line_potential([[.2,.3,.1]],f,self.lines,.75),line_potential([[.2,.3,.1]],self.f,self.lines,.75),atol=1e-12)
    def test_circle(self):
        angle=np.arange(8)*np.pi/4
        f={'robot_xy_m':np.column_stack([.75*np.cos(angle),.75*np.sin(angle)]),'confidence':[.9]*8}
        self.assertAlmostEqual(float(line_potential([[0,0,0]],f,[],.75)[0]),0.,places=12)
    def test_translation_and_rotation_equivariance(self):
        p=np.array([[.3,.5,.2]]);d=np.array([1.3,-.8]);a=.8
        R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        q=p.copy();q[:,:2]=q[:,:2]@R.T+d;q[:,2]+=a
        expected=line_potential(p,self.f,self.lines,.75)
        actual=line_potential(q,self.f,self.lines@R.T+d,.75,d)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    def test_mirror_equivalence(self):
        p=np.array([[.3,.5,.2],[-.3,-.5,.2+np.pi]])
        scores=line_potential(p,self.f,self.lines,.75)
        self.assertAlmostEqual(scores[0],scores[1],places=12)
    def test_gt_metadata_ignored(self):
        f=dict(self.f,gt_identity='nonsense',ground_truth_se2=[100,200,0])
        np.testing.assert_array_equal(line_potential([[.2,.4,.6]],f,self.lines,.75),line_potential([[.2,.4,.6]],self.f,self.lines,.75))
    def test_sampling_deterministic_bounded(self):
        p=np.ones((320,320))*.9
        a=support_pixels(p);b=support_pixels(p)
        self.assertEqual(a.shape,(32,3));np.testing.assert_array_equal(a,b)
        self.assertEqual(len(np.unique((a[:,:2]//20).astype(int),axis=0)),32)
    def test_invalid_particle_rejected(self):
        with self.assertRaises(ValueError):line_potential([[0,np.nan,0]],self.f,self.lines,.75)

if __name__=='__main__':unittest.main()
