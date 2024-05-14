from collections import namedtuple

DiffAndDistances = namedtuple("DiffAndDistances", "diff_el_el, dist_el_el, diff_el_ion, dist_el_ion, diff_ion_ion, dist_ion_ion, nonperiodic_diff_el_ion")
InputFeatures = namedtuple("InputFeatures", "el, ion, el_el, el_ion, ion_ion")
Embeddings = namedtuple("Embeddings", "el, ion, el_el, el_ion")
Edge = namedtuple("Edge", "tgt src")

WavefunctionDefinition = namedtuple("WavefunctionDefinition", "Z_max, Z_min")