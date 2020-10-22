import pytest
import numpy as np
import os
import shutil
import pysixtrack


aper_min_x = -0.04
aper_max_x = 0.03
aper_min_y = -0.02
aper_max_y = 0.01
aper_array = [[aper_max_x, aper_max_y],
              [aper_max_x, aper_min_y],
              [aper_min_x, aper_min_y],
              [aper_min_x, aper_max_y]]

mypolygon = np.array(aper_array).transpose()

poly_aper = pysixtrack.elements.LimitPolygon(aperture=mypolygon)
rect_aper = pysixtrack.elements.LimitRect(
    min_x=aper_min_x,
    max_x=aper_max_x,
    min_y=aper_min_y,
    max_y=aper_max_y
)

N_part = 20000


# -------------------------------------------------------
# ----Test scalar----------------------------------------
# -------------------------------------------------------
def test_scalar():
    p_scalar = pysixtrack.Particles()
    passed_particles_x_poly = []
    passed_particles_y_poly = []
    passed_particles_x_rect = []
    passed_particles_y_rect = []
    lost_particles_x_poly = []
    lost_particles_y_poly = []
    lost_particles_x_rect = []
    lost_particles_y_rect = []
    for n in range(N_part):
        p_scalar.x = (np.random.rand()-0.5) * 2.*8.5e-2
        p_scalar.y = (np.random.rand()-0.5) * 2.*8.5e-2
        p_scalar.state = 1

        ret = poly_aper.track(p_scalar)
        if p_scalar.state == 1:
            passed_particles_x_poly += [p_scalar.x]
            passed_particles_y_poly += [p_scalar.y]
        else:
            assert ret == "Particle lost"
            lost_particles_x_poly += [p_scalar.x]
            lost_particles_y_poly += [p_scalar.y]
        # check against LimitRect
        p_scalar.state = 1
        rect_aper.track(p_scalar)
        if p_scalar.state == 1:
            passed_particles_x_rect += [p_scalar.x]
            passed_particles_y_rect += [p_scalar.y]
        else:
            lost_particles_x_rect += [p_scalar.x]
            lost_particles_y_rect += [p_scalar.y]

    assert passed_particles_x_poly == passed_particles_x_rect
    assert passed_particles_y_poly == passed_particles_y_rect
    assert lost_particles_x_poly == lost_particles_x_rect
    assert lost_particles_y_poly == lost_particles_y_rect


# -------------------------------------------------------
# ----Test vector----------------------------------------
# -------------------------------------------------------
def test_vector():
    p_vec_poly = pysixtrack.Particles()
    p_vec_poly.x = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_vec_poly.y = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_vec_poly.state = np.ones_like(p_vec_poly.x, dtype=np.int)

    p_vec_rect = p_vec_poly.copy()

    poly_aper.track(p_vec_poly)
    rect_aper.track(p_vec_rect)

    assert np.array_equal(p_vec_poly.state, p_vec_rect.state)
    assert np.array_equal(p_vec_poly.x, p_vec_rect.x)
    assert np.array_equal(p_vec_poly.y, p_vec_rect.y)


def test_concave_poly():
    # We create and "H" shaped polygon and rotate it.
    # this way, we can cross check with 3 LimitRects and rotations
    p_vec_poly = pysixtrack.Particles()
    p_vec_poly.x = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_vec_poly.y = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_vec_poly.state = np.ones_like(p_vec_poly.x, dtype=np.int)
    p_vec_poly.partid = np.arange(len(p_vec_poly.x))
    p_vec_rect_left = p_vec_poly.copy()

    H_array = np.array([[2e-2, 1e-2],
                        [2e-2, 5e-2],
                        [3e-2, 5e-2],
                        [3e-2, -5e-2],
                        [2e-2, -5e-2],
                        [2e-2, -1e-2],
                        [-2e-2, -1e-2],
                        [-2e-2, -5e-2],
                        [-3e-2, -5e-2],
                        [-3e-2, 5e-2],
                        [-2e-2, 5e-2],
                        [-2e-2, 1e-2]]).transpose()
    rot_angle = 20. * np.pi/180.
    rot_matrix = [[np.cos(rot_angle), -1*np.sin(rot_angle)],
                  [np.sin(rot_angle), np.cos(rot_angle)]]
    H_aper_array = np.matmul(rot_matrix, H_array)
    poly_convex_aper = pysixtrack.elements.LimitPolygon(aperture=H_aper_array)

    rect_aper_left = pysixtrack.elements.LimitRect(
                            min_x=-3e-2,
                            max_x=-2e-2,
                            min_y=-5e-2,
                            max_y=5e-2
                        )
    rect_aper_mid = pysixtrack.elements.LimitRect(
                            min_x=-3e-2,
                            max_x=3e-2,
                            min_y=-1e-2,
                            max_y=1e-2
                        )
    rect_aper_right = pysixtrack.elements.LimitRect(
                            min_x=2e-2,
                            max_x=3e-2,
                            min_y=-5e-2,
                            max_y=5e-2
                        )
    rot_elem = pysixtrack.elements.SRotation(angle=rot_angle*180./np.pi)
    backrot_elem = pysixtrack.elements.SRotation(angle=-1*rot_angle*180./np.pi)

    rot_elem.track(p_vec_rect_left)
    p_vec_rect_mid = p_vec_rect_left.copy()
    p_vec_rect_right = p_vec_rect_left.copy()

    poly_convex_aper.track(p_vec_poly)
    rect_aper_left.track(p_vec_rect_left)
    backrot_elem.track(p_vec_rect_left)
    rect_aper_mid.track(p_vec_rect_mid)
    backrot_elem.track(p_vec_rect_mid)
    rect_aper_right.track(p_vec_rect_right)
    backrot_elem.track(p_vec_rect_right)

    # check if the surviving particles coincide
    for particle in p_vec_poly.partid:
        assert (particle in p_vec_rect_left.partid) \
            or (particle in p_vec_rect_mid.partid) \
            or (particle in p_vec_rect_right.partid)
    for particle in p_vec_rect_left.partid:
        assert particle in p_vec_poly.partid
    for particle in p_vec_rect_mid.partid:
        assert particle in p_vec_poly.partid
    for particle in p_vec_rect_right.partid:
        assert particle in p_vec_poly.partid


# -------------------------------------------------------
# ----Test mpmath compatibility--------------------------
# -------------------------------------------------------
def test_mpmath_compatibility():
    mpmath = pytest.importorskip("mpmath")
    mp = mpmath.mp

    p_mp = pysixtrack.Particles()
    mp.dps = 25
    p_mp.x = mp.mpf('3e-2') - mp.mpf('1e-27')
    p_mp.y = mp.mpf('1e-2')
    p_mp.state = 1
    polygon_mp = [[mp.mpf('-3e-2'), mp.mpf('3e-2'), mp.mpf('3e-2'), mp.mpf('-3e-2')],
                  [mp.mpf('4e-2'), mp.mpf('4e-2'), mp.mpf('-4e-2'), mp.mpf('-4e-2')]]
    aper_elem_mp = pysixtrack.elements.LimitPolygon(aperture=polygon_mp)

    aper_elem_mp.track(p_mp)

    assert p_mp.state == 1

    p_mp.x = mp.mpf('3e-2') + mp.mpf('1e-27')
    p_mp.y = mp.mpf('1e-2')
    p_mp.state = 1
    aper_elem_mp.track(p_mp)

    assert p_mp.state == 0


# -------------------------------------------------------
# ----Test MAD-X loader----------------------------------
# -------------------------------------------------------
def test_Polygon_mad_loader():
    madx = pytest.importorskip("cpymad.madx")

    madx = madx.Madx()
    madx.options.echo = False
    madx.options.warn = False
    madx.options.info = False

    tmpdir = './tmp/'
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    with open(tmpdir + 'test_poly_space.aper', 'w') as aper_file:
        for row in aper_array:
            aper_file.write(str(row[0]) + '   ' + str(row[1]) + '\n')
        aper_file.write(' \t ')
    with open(tmpdir + 'test_poly_tab.aper', 'w') as aper_file:
        for row in aper_array:
            aper_file.write(str(row[0]) + '\t' + str(row[1]) + '\n')

    madx.input('''
        TXQ1: Collimator, l=0.0, apertype='{}test_poly_space.aper';
        TXQ2: Collimator, l=0.0, apertype='{}test_poly_tab.aper';

        testseq: SEQUENCE, l=2.0;
            TXQ1, at = 1.0;
            TXQ2, at = 1.0;
        ENDSEQUENCE;

        BEAM, Particle=proton, Energy=50000.0, EXN=2.2e-6, EYN=2.2e-6;
        USE, Sequence=testseq;
    '''.format(tmpdir, tmpdir))

    seq = madx.sequence.testseq
    testline = pysixtrack.Line.from_madx_sequence(seq, install_apertures=True)
    poly_aper_mad = testline.elements[3]
    madx.input('stop;')

    p_poly_mad = pysixtrack.Particles()
    p_poly_mad.x = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_poly_mad.y = np.random.uniform(low=-8.5e-2, high=8.5e-2, size=N_part)
    p_poly_mad.state = np.ones_like(p_poly_mad.x, dtype=np.int)

    p_rect = p_poly_mad.copy()

    poly_aper_mad.track(p_poly_mad)
    rect_aper.track(p_rect)

    assert np.array_equal(p_poly_mad.state, p_rect.state)
    assert np.array_equal(p_poly_mad.x, p_rect.x)
    assert np.array_equal(p_poly_mad.y, p_rect.y)

    if os.path.isfile(tmpdir + 'test_poly_space.aper'):
        shutil.rmtree(tmpdir)
