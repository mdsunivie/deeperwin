#!/usr/bin/bash
date=$(date '+%Y-%m-%d')
deeperwin setup -i config_dpe_large_systems.yml -p experiment_name "reg_${date}_dpe" -p physical.name K -p comment rep1 rep2
deeperwin setup -i config_dpe_small_molecules.yml -p experiment_name "reg_${date}_dpe" -p physical.name NH3 Ethene N2_bond_breaking -p comment rep1 rep2
