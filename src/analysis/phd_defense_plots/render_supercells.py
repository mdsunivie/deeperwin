# %%
import subprocess
import os
import numpy as np
from deeperwin.run_tools.geometry_database import load_geometries, load_datasets, BOHR_IN_ANGSTROM
import scipy.spatial.transform

POVRAY_TEMPLATE = """#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic
  #right -4.54*x up 3.92*y
  #right -9.08*x up 7.84*y
  right -12*x up 10.361*y
  direction 1.00*z
  location <50.00,150.00,200.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  50.00> color White
  area_light <10, 0, 0>, <0, 10, 0>, 5, 5
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.070;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

ATOMS_PLACEHOLDER
"""


INI_TEMPLATE = """Input_File_Name=molecule.pov
Output_File_Name=OUTPUT_FILE_PLACEHOLDER.png
Output_to_File=True
Output_File_Type=N
Output_Alpha=on
; if you adjust Height, and width, you must preserve the ratio
; Width / Height = 1.157697
Width=925
Height=799.0000000000001
Antialias=True
Antialias_Threshold=0.1
Display=False
Pause_When_Done=True
Verbose=False
"""


def build_atoms_string(R, Z, color_scheme, material):
    radius = {1: 0.4}.get(Z, 0.7)
    color = color_scheme.get(Z, (40, 40, 40))
    color = [c / 256 for c in color]

    return f"atom( < {R[0]:.2f}, {R[1]:.2f}, {R[2]:.2f} >, {radius:.2f}, rgb < {color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f} >, 0.0, {material})"


def render(R, Z, name, material="glass"):
    output_dir = "/home/mscherbela/tmp/renders/"
    color_scheme = {1: (256, 256, 256), 6: (80, 80, 80), 7: (10, 10, 160), 8: (180, 10, 10)}

    with open(output_dir + "molecule.ini", "w") as f:
        f.write(INI_TEMPLATE.replace("OUTPUT_FILE_PLACEHOLDER", name))
    with open(output_dir + "molecule.pov", "w") as f:
        content = POVRAY_TEMPLATE.replace(
            "ATOMS_PLACEHOLDER", "\n".join([build_atoms_string(R, Z, color_scheme, material) for R, Z in zip(R, Z)])
        )
        f.write(content)
    subprocess.call(["povray", "molecule.ini"], cwd=output_dir)


def build_supercell(R, Z, n):
    n_total = int(np.prod(n))
    R_out = np.zeros_like(R, shape=(n_total * len(R), 3))
    Z_out = np.zeros_like(Z, shape=n_total * len(Z))
    offset = 0
    for nx in range(n[0]):
        for ny in range(n[1]):
            for nz in range(n[2]):
                R_out[offset : offset + len(R)] = R + np.array([nx, ny, nz])
                Z_out[offset : offset + len(Z)] = Z
                offset += len(R)
    return R_out, Z_out


R = np.array([[0, 0, 0], [0.5, 0.5, -0.5]])
Z = [8, 1]
scale = 1.6

for material in ["glass", "jmol"]:
    for n in range(1, 5):
        R_sc, Z_sc = build_supercell(R, Z, [n, n, n])
        R_sc -= np.mean(R_sc, axis=0)
        render(R_sc * scale, Z_sc, f"test_{material}_{n}", material)
