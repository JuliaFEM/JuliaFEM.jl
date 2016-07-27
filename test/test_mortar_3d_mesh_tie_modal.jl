# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM
using JuliaFEM.Preprocess
using JuliaFEM.Postprocess
using JuliaFEM.Testing

#=
test subjects:
- modal analysis, with mesh tie contact

Fixed-fixed solution is ωᵢ = λᵢ²√(EI/ρA) , where λᵢ = cosh(λᵢℓ)cos(λᵢℓ)

1:  4.730040744862704
2:  7.853204624095838
3: 10.995607838001671

[1] De Silva, Clarence W. Vibration: fundamentals and practice. CRC press, 2006, p.355

Code Aster solution:
--------------------
numéro    fréquence (HZ)     norme d'erreur
    1       1.12946E+00        5.81018E-12
    2       1.13141E+00        6.33463E-12
    3       2.93779E+00        6.53408E-13
    4       2.94143E+00        5.43970E-13
    5       4.51684E+00        5.43252E-13
=#


comm_CA = """
DEBUT(PAR_LOT="NON")

MAIL = LIRE_MAILLAGE(FORMAT="MED", NOM_MED="CYLINDER_20_SPLITTED")

MO = AFFE_MODELE(
    MAILLAGE=MAIL,
    AFFE=_F(TOUT="OUI",
            PHENOMENE="MECANIQUE", MODELISATION="3D"))

MAT = DEFI_MATERIAU(
    ELAS=_F(E=50475.45, NU=0.3, RHO=1.0))

CHMAT = AFFE_MATERIAU(
    MAILLAGE=MAIL,
    AFFE=_F(TOUT="OUI", MATER=MAT))

BC1 = AFFE_CHAR_MECA(
    MODELE=MO,
    DDL_IMPO=(
        _F(GROUP_MA=("CYLINDER_20_1_FACE1"), DX=0, DY=0, DZ=0)))

BC2 = AFFE_CHAR_MECA(
    MODELE=MO,
    DDL_IMPO=(
        _F(GROUP_MA=("CYLINDER_20_2_FACE2"), DX=0, DY=0, DZ=0)))

# ESCL = SLAVE
# MAIT = MASTER
BC3 = AFFE_CHAR_MECA(
    MODELE=MO,
    LIAISON_MAIL=_F(
        GROUP_MA_ESCL="CYLINDER_20_1_FACE2",
        GROUP_MA_MAIT="CYLINDER_20_2"))

# assemble material stiffness matrix

RIGEL = CALC_MATR_ELEM(
    MODELE=MO,
    OPTION="RIGI_MECA",
    CHAM_MATER=CHMAT,
    CHARGE=(BC1, BC2, BC3))

NUMEDDL = NUME_DDL(
    MATR_RIGI=RIGEL)

RIGAS = ASSE_MATRICE(
    MATR_ELEM=RIGEL,
    NUME_DDL=NUMEDDL)

# assemble mass matrix

MASSEL = CALC_MATR_ELEM(
    MODELE=MO,
    OPTION="MASS_MECA",
    CHAM_MATER=CHMAT,
    CHARGE=(BC1, BC2, BC3))

MASSAS = ASSE_MATRICE(
    MATR_ELEM=MASSEL,
    NUME_DDL=NUMEDDL)

# modal analysis, without geometric stiffness

BRESU = CALC_MODES(
    MATR_RIGI=RIGAS,
    MATR_MASS=MASSAS,
    OPTION="BANDE",
    CALC_FREQ=_F(
        FREQ=(0.0, 5.0)))

# modal analysis, with geometric stiffness

BRESU = NORM_MODE(
    reuse=BRESU,
    MODE=BRESU,
    NORME="TRAN")

IMPR_RESU(
    MODELE=MO,
    FORMAT="RESULTAT",
    RESU=_F(RESULTAT=BRESU))

IMPR_RESU(
    FORMAT="MED",
    UNITE=80,
    RESU=_F(RESULTAT=BRESU))

FIN()
"""

@testset "splitted rod with tie contact" begin
    # CYLINDER_20_1_FACE1 -- CYLINDER_20_1_FACE2 -- CYLINDER_20_2_FACE_1 -- CYLINDER_20_2_FACE_2
    mesh_file = Pkg.dir("JuliaFEM") * "/test/testdata/primitives.med"
    mesh = aster_read_mesh(mesh_file, "CYLINDER_20_SPLITTED")
    body1 = Problem(mesh, Elasticity, "CYLINDER_20_1", 3)
    body2 = Problem(mesh, Elasticity, "CYLINDER_20_2", 3)
    for body in [body1, body2]
        update!(body.elements, "youngs modulus", 54475.45)
        update!(body.elements, "poissons ratio", 0.3)
        update!(body.elements, "density", 1.0)
    end
    bc1 = Problem(mesh, Dirichlet, "CYLINDER_20_1_FACE1", 3, "displacement")
    bc2 = Problem(mesh, Dirichlet, "CYLINDER_20_2_FACE2", 3, "displacement")
    for bc in [bc1, bc2]
        update!(bc.elements, "displacement 1", 0.0)
        update!(bc.elements, "displacement 2", 0.0)
        update!(bc.elements, "displacement 3", 0.0)
    end
    interface = Problem(Mortar, "interface between bodies", 3, "displacement")
    slave = create_elements(mesh, "CYLINDER_20_1_FACE2")
    master = create_elements(mesh, "CYLINDER_20_2_FACE1")
    update!(slave, "master elements", master)
    interface.elements = [slave; master]

    solver = Solver(Modal, body1, body2, bc1, bc2, interface)
    solver.properties.nev = 5
    solver.properties.which = :SM
    solver()
    freqs_jf = sqrt(solver.properties.eigvals)/(2*pi)

    freqs_ca = [1.12946E+00, 1.13141E+00, 2.93779E+00, 2.94143E+00, 4.51684E+00]
    freq_jf = freqs_jf[1]
    freq_ca = freqs_ca[1]
    rtol = norm(freq_jf - freq_ca)/max(freq_jf, freq_ca)
    info("rtol = $rtol")
    for (i, freq) in enumerate(freqs_jf)
        @printf "mode %i | freq JuliaFEM %8.3f | freq Code Aster %8.3f\n" i freqs_jf[i] freqs_ca[i]
    end
    if rtol > 1.0e-3
        outfile = tempname() * ".xmf"
        info("Something went wrong, results are saved to $outfile")
        result = XDMF()
        elems = [body1.elements; body2.elements]
        for (i, freq) in enumerate(freqs_jf)
            xdmf_new_result!(result, elems, freq)
            xdmf_save_field!(result, elems, freq, "displacement"; field_type="Vector")
        end
        xdmf_save!(result, outfile)
    end
    @test rtol < 0.05

end

