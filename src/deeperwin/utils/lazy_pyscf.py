_pyscf = None


def __getattr__(name):
    global _pyscf
    if _pyscf is None:
        import pyscf
        import pyscf.fci
        import pyscf.pbc.gto
        import pyscf.ci
        import pyscf.lo
        import pyscf.mcscf

        _pyscf = pyscf
    return getattr(_pyscf, name)
