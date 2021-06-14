import pandas as pd
import io

EXACT_ENERGIES = """System;E_HF;E_exact;E_corr;Bond_length;Spin_state;Source
He;-2.86168;-2.90372;0.04204;;;CCSD(T) benchmarks https://doi.org/10.1063/1.4798707
Li;-7.43275;-7.47806;0.04531;;;FermiNet Paper
Be;-14.57301;-14.66736;0.09435;;;FermiNet Paper
B;-24.53316;-24.65391;0.12075;;;FermiNet Paper
C;-37.69380;-37.84500;0.15120;;;FermiNet Paper
N;-54.40470;-54.58920;0.18450;;;FermiNet Paper
O;-74.81920;-75.06730;0.24810;;;FermiNet Paper
F;-99.41680;-99.73390;0.31710;;;FermiNet Paper
Ne;-128.54790;-128.93760;0.38970;;;FermiNet Paper
H2;-1.13362;-1.17448;0.04086;1.40108;;CCSD(T) benchmarks https://doi.org/10.1063/1.4798707
LiH;-7.98737;-8.07055;0.08318;3.01500;;FermiNet Paper
N2;-108.99400;-109.54230;0.54830;2.06800;;FermiNet Paper
Li2;-14.87155;-14.99540;0.12385;5.05100;;FermiNet Paper
CO;-112.78710;-113.32550;0.53840;2.17300;;FermiNet Paper (CCSD(T))
Be2;-29.13408;-29.33800;0.20392;4.65000;;Morales VMC/DMC extrapolation, HF from https://doi.org/10.1063/1.4798707
B2;-49.09010;-49.41410;0.32400;3.00500;3Sigma-;Morales VMC/DMC extrapolation
C2;-75.40666;-75.92650;0.51984;2.34810;;Morales VMC/DMC extrapolation
O2;-149.66875;-150.32740;0.65865;2.28300;3Sigma-;Morales VMC/DMC extrapolation
F2;-198.77352;-199.53040;0.75688;2.66800;;Morales VMC/DMC extrapolation
H4Rect;;-2.0155;;R=3.2843, theta=90deg;;FermiNet Paper
H3plus;;-1.343835518;;1.65;;Cencek 1998 American Institute of Physics. @S0021-9606(98)00906-4
H4plus;;-1.852733;;;;2008, DOI: 10.1063/1.2953571
HChain6;;-3.37619;;R=1.8;;https://doi.org/10.1021/acs.jpclett.7b00689 (Not necessary hundred percent exact)
HChain10;;-5.6655;;R=1.801;;FermitNet Paper https://doi.org/10.1103/PhysRevX.7.031059"""

BENCHMARK_DB = """System;Author;Method;N_Determinants;Epochs;Energy;E_exact;Error_eval
LiH;Berlin;SD-SJBF;1;1000;-8.06989222035022;-8.070548;1
Be;Berlin;SD-SJBF;1;1000;-14.6615344671349;-14.66736;6
B;Berlin;SD-SJBF;1;1000;-24.6417503494892;-24.65391;12
LiH;Berlin;SD-SJBF;1;9999;-8.07005978849465;-8.070548;0
Be;Berlin;SD-SJBF;1;9999;-14.6612252893739;-14.66736;6
B;Berlin;SD-SJBF;1;9999;-24.6427544632751;-24.65391;11
LiH;Berlin;MD-SJBF;;1000;-8.07007183055805;-8.070548;0
Be;Berlin;MD-SJBF;;1000;-14.6670053551893;-14.66736;0
B;Berlin;MD-SJBF;;1000;-24.6491482649404;-24.65391;5
LiH;Berlin;MD-SJBF;;9999;-8.07036270313129;-8.070548;0
Be;Berlin;MD-SJBF;;9999;-14.6672380080854;-14.66736;0
B;Berlin;MD-SJBF;;9999;-24.6505561753491;-24.65391;3
LiH;Berlin;SD-SJBF;1;;-8.06997489929199;-8.070548;1
Be;Berlin;SD-SJBF;1;;-14.6604976654053;-14.66736;7
B;Berlin;SD-SJBF;1;;-24.6420631408691;-24.65391;12
C;Berlin;SD-SJBF;1;;-37.8306617736816;-37.845;14
LiH;Berlin;MD-SJBF;;;-8.07027244567871;-8.070548;0
Be;Berlin;MD-SJBF;;;-14.6672897338867;-14.66736;0
B;Berlin;MD-SJBF;;;-24.6507244110107;-24.65391;3
C;Berlin;MD-SJBF;;;-37.8367462158203;-37.845;8
B2;Berlin;SD-SJBF;1;;-49.3844909667969;-49.4141;30
B2;Berlin;MD-SJBF;3;;-49.3915100097656;-49.4141;23
B2;Berlin;MD-SJBF;10;;-49.400505065918;-49.4141;14
B2;Berlin;MD-SJBF;30;;-49.4049758911133;-49.4141;9
B2;Berlin;MD-SJBF;100;;-49.4057388305664;-49.4141;8
Be2;Berlin;SD-SJBF;1;;-29.3208541870117;-29.338;17
Be2;Berlin;MD-SJBF;3;;-29.3224601745605;-29.338;16
Be2;Berlin;MD-SJBF;10;;-29.3304233551025;-29.338;8
Be2;Berlin;MD-SJBF;30;;-29.3309116363525;-29.338;7
Be2;Berlin;MD-SJBF;100;;-29.3352394104004;-29.338;3
C2;Berlin;SD-SJBF;1;;-75.8696823120117;-75.9265;57
C2;Berlin;MD-SJBF;3;;-75.8830184936523;-75.9265;43
C2;Berlin;MD-SJBF;10;;-75.8895416259766;-75.9265;37
C2;Berlin;MD-SJBF;30;;-75.8946380615234;-75.9265;32
C2;Berlin;MD-SJBF;100;;-75.9038925170898;-75.9265;23
Li2;Berlin;SD-SJBF;1;;-14.9907674789429;-14.9954;5
Li2;Berlin;MD-SJBF;3;;-14.9926853179932;-14.9954;3
Li2;Berlin;MD-SJBF;10;;-14.9933567047119;-14.9954;2
Li2;Berlin;MD-SJBF;30;;-14.9943313598633;-14.9954;1
Li2;Berlin;MD-SJBF;100;;-14.9942588806152;-14.9954;1"""

def get_reference_energies(get_full_df = False):
    """
    Return quasi-exact reference energies (in Hartree) for groundstates of various small molecules/atoms.

    Args:
        get_full_df (bool): If true, returns the reference energies as a pandas dataframe, including the fields 'System', 'E_exact' and auxiliary fields. If false, only a dict with key=System, value=E_exact is returned.

    Returns:
        (dict or pd.DataFrame): Groundstate energies

    """
    if not hasattr(get_reference_energies, 'df'):
        get_reference_energies.df = pd.read_csv(io.StringIO(EXACT_ENERGIES), delimiter=';')
        get_reference_energies.data_dict = {k: v for k, v in
                                            zip(get_reference_energies.df.System, get_reference_energies.df.E_exact)}
    if get_full_df:
        return get_reference_energies.df
    else:
        return get_reference_energies.data_dict

def get_benchmark_db():
    """
    Return (non-exhaustive) list of results obtained for small molecules by other neural network based VMC codes.

    Returns:
        (pd.DataFrame): Table containing the following fields: System;Author;Method;N_Determinants;Epochs;Energy;E_exact;Error_eval

    """
    if not hasattr(get_benchmark_db, 'df'):
        get_benchmark_db.df = pd.read_csv(io.StringIO(BENCHMARK_DB), delimiter=';')
    return get_benchmark_db.df


if __name__ == '__main__':
    d = get_reference_energies(False)
    print(d['H4Rect'])
    # df = get_benchmark_db()
    # ref = df[df.Author=='Berlin'].groupby('System').min()['Energy']

