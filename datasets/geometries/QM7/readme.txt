
Data set dsgdb7ae
=================

DFT atomization energies for 7k small organic molecules.

Please cite these publications if you use this data set:
* Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert M"uller, O. Anatole von 
  Lilienfeld: Fast and Accurate Modeling of Molecular Atomization Energies with 
  Machine Learning, Physical Review Letters, 108(5): 058301, 2012. 
  DOI: 10.1103/PhysRevLett.108.058301
* Gr\'egoire Montavon, Katja Hansen, Siamac Fazli, Matthias Rupp, Franziska
  Biegler, Andreas Ziehe, Alexandre Tkatchenko, O. Anatole von Lilienfeld,
  Klaus-Robert M"uller: Learning Invariant Representations of Molecules for
  Atomization Energy Prediction, Advances in Neural Information Processing 
  Systems 25 (NIPS 2012), Lake Tahoe, Nevada, USA, December 3-6, 2012.

A version of this data set with pre-calculated Coulomb matrices is available at
http://quantum-machine.org (last accessed 2013-01-15).

Files
-----

dsgdb7ae.xyz          - Molecules and atomization energies in XYZ format.
dsgdb7ae_cvsplits.txt - Indices (1-based) of 10 repetitions of 5-fold 
                        stratified cross-validation.
dsgdb7ae_subset1k.txt - Indices of stratified subset of 1000 molecules.
readme.txt            - Documentation.

Molecules
---------

A subset of 7165 small organic molecules from the GDB-13 database [1]. It
contains all molecules with up to 7 non-hydrogen atoms and elements H,C,N,O,S.
Molecular geometries were relaxed using the universal force field [2] as
implemented in OpenBabel [3]. Coordinates are in Angstrom; to convert to 
atomic units, multiply by 100/52.917720859.

Atomization energies
--------------------

Atomization energies were calculated at the Density Functional Theory level
using the PBE0 functional. Units are kcal/mol. Divide by 23.045108 to convert
to eV.

Cross-validation splits
-----------------------

Indices (starting from 1) for 10 repetitions of 5-fold stratified cross-
validation are provided. Stratification is by energy so that each fold covers
the whole energy range.

1k subset
---------

Indices (starting from 1) for a stratified subset of 1000 molecules are 
provided. Stratification is by energy so that the subset covers the whole
energy range.

References
----------

[1] Lorenz C. Blum, Jean-Louis Reymond: 970 Million Druglike Small Molecules
    for Virtual Screening in the Chemical Universe Database GDB-13, Journal of
    the American Chemical Society 131(25): 8732-8733, 2009.
[2] Anthony K. Rapp\'e, Carla J. Casewit, K. S. Colwell, William A. Goddard III,
    W. Mason Skiff: UFF, a full periodic table force field for molecular 
    mechanics and molecular dynamics simulations, Journal of the American
    Chemical Society 114(25): 10024-10035, 1992.
[3] Rajarshi Guha, Michael T. Howard, Geoffrey R. Hutchison, Peter Murray-Rust,
    Henry Rzepa, Christoph Steinbeck, J"org Wegner, Egon L. Willighagen: The 
    Blue Obelisk - Interoperability in Chemical Informatics, Journal of 
    Chemical Information and Modeling 46(3): 991-998, 2006.
