# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

using JuliaFEM, Test

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

# CYLINDER_20_1_FACE1 -- CYLINDER_20_1_FACE2 -- CYLINDER_20_2_FACE_1 -- CYLINDER_20_2_FACE_2

mesh_file = @__DIR__() * "/testdata/primitives.med"
mesh = aster_read_mesh(mesh_file, "CYLINDER_20_SPLITTED")
body1 = Problem(Elasticity, "CYLINDER_20_1", 3)
body2 = Problem(Elasticity, "CYLINDER_20_2", 3)
body1_elements = create_elements(mesh, "CYLINDER_20_1")
body2_elements = create_elements(mesh, "CYLINDER_20_2")
for element_set in [body1_elements, body2_elements]
    update!(element_set, "youngs modulus", 54475.45)
    update!(element_set, "poissons ratio", 0.3)
    update!(element_set, "density", 1.0)
end
add_elements!(body1, body1_elements)
add_elements!(body2, body2_elements)

bc1 = Problem(Dirichlet, "CYLINDER_20_1_FACE1", 3, "displacement")
bc2 = Problem(Dirichlet, "CYLINDER_20_2_FACE2", 3, "displacement")
bc1_elements = create_elements(mesh, "CYLINDER_20_1_FACE1")
bc2_elements = create_elements(mesh, "CYLINDER_20_2_FACE2")
for element_set in [bc1_elements, bc2_elements]
    update!(element_set, "displacement 1", 0.0)
    update!(element_set, "displacement 2", 0.0)
    update!(element_set, "displacement 3", 0.0)
end
add_elements!(bc1, bc1_elements)
add_elements!(bc2, bc2_elements)

interface = Problem(Mortar, "interface between bodies", 3, "displacement")
slave = create_elements(mesh, "CYLINDER_20_1_FACE2")
master = create_elements(mesh, "CYLINDER_20_2_FACE1")
update!(slave, "master elements", master)
interface.elements = [slave; master]

analysis = Analysis(Modal)
add_problems!(analysis, body1, body2, bc1, bc2, interface)
analysis.properties.nev = 5
analysis.properties.which = :SM
run!(analysis)
freqs_jf = sqrt.(analysis.properties.eigvals)/(2*pi)
freqs_ca = [1.12946E+00, 1.13141E+00, 2.93779E+00, 2.94143E+00, 4.51684E+00]
@test isapprox(freqs_ca, freqs_jf; rtol=0.04)
