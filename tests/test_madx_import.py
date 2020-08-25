import os
import sys
from importlib import util

import pysixtrack
import pysixtrack.be_beamfields.tools as bt


def test_madx_import():
    cpymad_spec = util.find_spec("cpymad")
    if cpymad_spec is None:
        print("cpymad is not available - abort test")
        sys.exit(0)

    from cpymad.madx import Madx

    seq_name = "psb1"
    use_aperture = True

    n_SCkicks = 120
    length_fuzzy = 0.0
    p0c = 0.571e6
    particle = pysixtrack.Particles(p0c=p0c)
    betagamma = particle.beta0 * particle.gamma0
    # mass = pysixtrack.Particles.pmass
    delta_rms = 1e-3
    neps_x = 1.5e-6
    neps_y = 1.5e-6

    # for space charge bunched
    number_of_particles = 1e11
    bunchlength_rms = 1.0

    # for space charge coasting
    line_density = 1e11

    for sc_mode in ["Bunched", "Coasting"]:

        mad = Madx()
        mad.options.echo = False
        mad.options.info = False
        mad.warn = False
        file_path = os.path.realpath(__file__)
        path = os.path.dirname(file_path) + "/psb/"
        mad.call(path + "psb_fb_lhc.madx", chdir=True)

        # Determine space charge locations
        temp_line = pysixtrack.Line.from_madx_sequence(mad.sequence[seq_name])
        sc_locations, sc_lengths = bt.determine_sc_locations(
            temp_line, n_SCkicks, length_fuzzy
        )

        # Install spacecharge place holders
        sc_names = ["sc%d" % number for number in range(len(sc_locations))]
        bt.install_sc_placeholders(
            mad, seq_name, sc_names, sc_locations, mode=sc_mode
        )

        # Generate line with spacecharge
        line = pysixtrack.Line.from_madx_sequence(
            mad.sequence[seq_name], install_apertures=use_aperture
        )

        # Get sc info from optics
        mad_sc_names, sc_twdata = bt.get_spacecharge_names_twdata(
            mad, seq_name, mode=sc_mode
        )

        # Check consistency
        if sc_mode == "Bunched":
            sc_elements, sc_names = line.get_elements_of_type(
                pysixtrack.elements.SpaceChargeBunched
            )
        elif sc_mode == "Coasting":
            sc_elements, sc_names = line.get_elements_of_type(
                pysixtrack.elements.SpaceChargeCoasting
            )
        else:
            raise ValueError("mode not understood")
        bt.check_spacecharge_consistency(
            sc_elements, sc_names, sc_lengths, mad_sc_names
        )

        # Setup spacecharge in the line
        if sc_mode == "Bunched":
            bt.setup_spacecharge_bunched_in_line(
                sc_elements,
                sc_lengths,
                sc_twdata,
                betagamma,
                number_of_particles,
                bunchlength_rms,
                delta_rms,
                neps_x,
                neps_y,
            )
        elif sc_mode == "Coasting":
            bt.setup_spacecharge_coasting_in_line(
                sc_elements,
                sc_lengths,
                sc_twdata,
                betagamma,
                line_density,
                delta_rms,
                neps_x,
                neps_y,
            )
        else:
            raise ValueError("mode not understood")


def test_error_import():
    cpymad_spec = util.find_spec("cpymad")
    if cpymad_spec is None:
        print("cpymad is not available - abort test")
        sys.exit(0)

    from cpymad.madx import Madx
    
    madx = Madx()
    
    madx.input('''
        MQ1: Quadrupole, K1:=KQ1, L=1.0, apertype=CIRCLE, aperture={0.04};
        MQ2: Quadrupole, K1:=KQ2, L=1.0, apertype=CIRCLE, aperture={0.04};
        MQ3: Quadrupole, K1:=0.0, L=1.0, apertype=CIRCLE, aperture={0.04};

        KQ1 = 0.02;
        KQ2 = -0.02;

        testseq: SEQUENCE, l = 20.0;
            MQ1, at =  5;
            MQ2, at = 12;
            MQ3, at = 18;
        ENDSEQUENCE;

        !---the usual stuff
        BEAM, PARTICLE=PROTON, ENERGY=7000.0, EXN=2.2e-6, EYN=2.2e-6;
        USE, SEQUENCE=testseq;


        Select, flag=makethin, pattern="MQ1", slice=2;
        makethin, sequence=testseq;

        use, sequence=testseq;

        !---misalign collimator
        select, flag = error, clear;
        select, flag = error, pattern = "MQ1";
        ealign, dx = 0.01, dy = 0.01, arex = 0.02, arey = 0.02;
        select, flag = error, clear;
        select, flag = error, pattern = "MQ2";
        ealign, dx = 0.04, dy = 0.04, dpsi = 0.1;
        select, flag = error, clear;
        select, flag = error, pattern = "MQ3";
        ealign, dx = 0.00, dy = 0.00, arex = 0.00, arey = 0.00, dpsi = 0.00;
        efcomp, DKN = {0.0, 0.0, 0.001, 0.002}, DKS = {0.0, 0.0, 0.003, 0.004};
        select, flag = error, full;
    ''')
    seq = madx.sequence.testseq
    # store already applied errors:
    madx.command.esave(file='lattice_errors.err')
    madx.command.readtable(file='lattice_errors.err', table="errors")
    os.remove('lattice_errors.err')
    errors = madx.table.errors

    pysixtrack_line = pysixtrack.Line.from_madx_sequence(seq, install_apertures=True)
    pysixtrack_line.apply_madx_errors(errors)
    madx.input('stop;')
    
    expected_element_num = ( 2 # start and end marker
                            +6 # drifts (including drift between MQ1 slices)
                            +3+2 # quadrupoles + MQ1 slices
                            +3+2 # corresponding aperture elements
                            +2*(3+1) # dx/y in/out for MQ1 slices and MQ2
                            +2 # tilt in/out for MQ2
                            +2*3 # arex/y in/out for MQ1 slices
                           )
    assert len(pysixtrack_line) == expected_element_num
    
    expected_element_order = [pysixtrack.elements.Drift,    # start marker
                              pysixtrack.elements.Drift,
                              pysixtrack.elements.XYShift,  # dx/y in of MQ1 1st slice
                              pysixtrack.elements.Multipole,# MQ1 1st slice
                              pysixtrack.elements.XYShift,  # arex/y in for MQ1 1st slice
                              pysixtrack.elements.LimitEllipse,# MQ1 1st slice aperture
                              pysixtrack.elements.XYShift,  # arex/y out for MQ1 1st slice
                              pysixtrack.elements.XYShift,  # dx/y out for MQ1 1st slice
                              pysixtrack.elements.Drift,
                              pysixtrack.elements.XYShift,  # dx/y in for MQ1 marker
                              pysixtrack.elements.Drift,    # MQ1 marker
                              pysixtrack.elements.XYShift,  # arex/y in for MQ1 marker
                              pysixtrack.elements.LimitEllipse,# MQ1 marker aperture
                              pysixtrack.elements.XYShift,  # arex/y out for MQ1 marker
                              pysixtrack.elements.XYShift,  # dx/y out for MQ1 marker
                              pysixtrack.elements.Drift,
                              pysixtrack.elements.XYShift,  # dx/y in for MQ1 2nd slice
                              pysixtrack.elements.Multipole,# MQ1 2nd slice
                              pysixtrack.elements.XYShift,  # arex/y in for MQ1 2nd slice
                              pysixtrack.elements.LimitEllipse,# MQ1 2nd slice aperture
                              pysixtrack.elements.XYShift,  # arex/y out for MQ1 2nd slice
                              pysixtrack.elements.XYShift,  # dx/y out for MQ1 2nd slice
                              pysixtrack.elements.Drift,
                              pysixtrack.elements.XYShift,  # dx/y in for MQ2
                              pysixtrack.elements.SRotation,# tilt in for MQ2
                              pysixtrack.elements.Multipole,# MQ2
                              pysixtrack.elements.LimitEllipse,# MQ2 aperture
                              pysixtrack.elements.SRotation,# tilt out for MQ2
                              pysixtrack.elements.XYShift,  # dx/y out for MQ2
                              pysixtrack.elements.Drift,
                              pysixtrack.elements.Multipole,# MQ3
                              pysixtrack.elements.LimitEllipse,# MQ3 aperture
                              pysixtrack.elements.Drift,
                              pysixtrack.elements.Drift     # end marker
                             ]
    for element, expected_element in zip(pysixtrack_line.elements,
                                         expected_element_order):
        assert isinstance(element,expected_element)

    idx_MQ3 = pysixtrack_line.element_names.index('mq3')
    MQ3 = pysixtrack_line.elements[idx_MQ3]
    assert abs(MQ3.knl[2] - 0.001) < 1e-14
    assert abs(MQ3.knl[3] - 0.002) < 1e-14
    assert abs(MQ3.ksl[2] - 0.003) < 1e-14
    assert abs(MQ3.ksl[3] - 0.004) < 1e-14


def test_neutral_errors():
    # make sure that some misaligned drifts do not influence particle
    cpymad_spec = util.find_spec("cpymad")
    if cpymad_spec is None:
        print("cpymad is not available - abort test")
        sys.exit(0)

    from cpymad.madx import Madx
    
    madx = Madx()
    
    madx.input('''
        T1: Collimator, L=1.0, apertype=CIRCLE, aperture={0.5};
        T2: Collimator, L=1.0, apertype=CIRCLE, aperture={0.5};
        T3: Collimator, L=1.0, apertype=CIRCLE, aperture={0.5};

        KQ1 = 0.02;
        KQ2 = -0.02;

        testseq: SEQUENCE, l = 20.0;
            T1, at =  5;
            T2, at = 12;
            T3, at = 18;
        ENDSEQUENCE;

        !---the usual stuff
        BEAM, PARTICLE=PROTON, ENERGY=7000.0, EXN=2.2e-6, EYN=2.2e-6;
        USE, SEQUENCE=testseq;


        Select, flag=makethin, pattern="MQ1", slice=2;
        makethin, sequence=testseq;

        use, sequence=testseq;

        !---misalign collimator
        select, flag = error, clear;
        select, flag = error, pattern = "T1";
        ealign, dx = 0.01, dy = 0.01, arex = 0.02, arey = 0.02;
        select, flag = error, clear;
        select, flag = error, pattern = "T2";
        ealign, dx = 0.04, dy = 0.04, dpsi = 0.1;
        select, flag = error, clear;
        select, flag = error, pattern = "T3";
        ealign, dx = 0.02, dy = 0.01, arex = 0.03, arey = 0.02, dpsi = 0.1;
        select, flag = error, full;
    ''')
    seq = madx.sequence.testseq
    # store already applied errors:
    madx.command.esave(file='lattice_errors.err')
    madx.command.readtable(file='lattice_errors.err', table="errors")
    os.remove('lattice_errors.err')
    errors = madx.table.errors

    pysixtrack_line = pysixtrack.Line.from_madx_sequence(seq, install_apertures=True)
    pysixtrack_line.apply_madx_errors(errors)
    madx.input('stop;')
    
    initial_x = 0.025
    initial_y = -0.015
    
    particle = pysixtrack.Particles()
    particle.x = initial_x
    particle.y = initial_y
    particle.state = 1
    
    pysixtrack_line.track(particle)
    
    assert abs(particle.x-initial_x) < 1e-14
    assert abs(particle.y-initial_y) < 1e-14
    
