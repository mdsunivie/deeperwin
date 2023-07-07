"""
Haiku NN modules representing orbital functions
"""

from typing import Tuple


def _determine_n_output_orbitals(
    n_up: int,
    n_dn: int,
    determinant_schema: str
) -> Tuple[int]:
    """
    Function to determine the dimensionality of the outputted up & down MO matrix blocks given
    the number of up & dn electrons.
    """
    if determinant_schema == "full_det":
        return (n_up + n_dn, n_up + n_dn)
    elif determinant_schema == "block_diag":
        return (n_up, n_dn)
    elif determinant_schema == "restricted_closed_shell":
        return (n_up, n_dn)

def _determine_elec_idxs(
    n_up: int,
    n_dn: int,
    determinant_schema: str
) -> Tuple[int]:
    """
    Function to determine indices for indexing electron embeddings when inferring the
    MO matrix blocks
    """
    if determinant_schema == "full_det":
        return (n_up, n_dn)
    elif determinant_schema == "block_diag":
        return (n_up, n_dn)
    elif determinant_schema == "restricted_closed_shell":
        return (n_up + n_dn, n_up + n_dn)


