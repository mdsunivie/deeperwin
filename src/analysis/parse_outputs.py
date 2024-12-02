import parse
import pandas as pd
import os

REFERENCE_ENERGIES = {'He': -2.90372, 'Li': -7.47806, 'Be': -14.66736, 'B': -24.65391, 'C': -37.845, 'N': -54.5892, 'O': -75.0673, 'F': -99.7339, 'Ne': -128.9376, 'H2': -1.17448, 'LiH': -8.07055, 'N2': -109.5423, 'Li2': -14.9954, 'CO': -113.3255, 'Be2': -29.338, 'B2': -49.4141, 'C2': -75.9265, 'O2': -150.3274, 'F2': -199.5304, 'H4Rect': -2.0155, 'H3plus': -1.3438355180000001, 'H4plus': -1.8527330000000002, 'HChain6': -3.3761900000000002, 'HChain10': -5.6655}
REFERENCE_ENERGIES['H10'] = REFERENCE_ENERGIES['HChain10']

def parse_logfile(fname):
    with open(fname) as f:
        data = {}
        for line in f:
            line = line.strip()
            if 'Full config: ' in line:
                config = eval(line.split('Full config: ')[-1])
                data.update(config)
            elif 'Using the following settings: ' in line:
                config = eval(line.split("Using the following settings: ")[-1])
                data.update(config)
            elif 'Number of parameters: ' in line:
                data['n_params'] = int(line.split("Number of parameters: ")[-1])
            elif "Used hardware: gpu" in line:
                data['hardware'] = 'gpu'
            elif "Used hardware: cpu" in line:
                data['hardware'] = 'cpu'
            elif 'Eval per step' in line:
                data['t_eval'] = parse.parse("{}Eval per step        : {:8.4f} sec", line)[1]
            elif 'Optimization per step' in line:
                data['t_opt'] = parse.parse("{}Optimization per step: {:8.4f} sec", line)[1]
            elif 'Tracing optimization :' in line:
                data['t_tracing_opt'] = parse.parse("{}Tracing optimization : {:8.4f} sec", line)[1]
            elif 'Tracing evaluation   :' in line:
                data['t_tracing_eval'] = parse.parse("{}Tracing evaluation   : {:8.4f} sec", line)[1]
            elif 'Energy QMC           :' in line:
                data['E_eval'], data['sigma_E_eval'] = parse.parse("{}Energy QMC           : {:.4f} +- {:.4f}Ha", line)[1:]

    if 'molecule' in data and data['molecule'] in REFERENCE_ENERGIES:
        data['E_ref'] = REFERENCE_ENERGIES[data['molecule']]
        if 'E_eval' in data:
            data['error_eval'] = 1e3*(data['E_eval'] - data['E_ref'])
    job_fname = fname.replace('GPU.out', 'job.sh')
    data['gpu'] = None
    data['directory'] = os.path.dirname(fname)
    data['directory_short'] = "/".join(data['directory'].split('/')[-2:])
    if os.path.isfile(job_fname):
        with open(job_fname) as f:
            for line in f:
                if '#SBATCH --qos gpu_' in line:
                    data['gpu'] = line.strip().split('#SBATCH --qos gpu_')[-1]

    return data

def read_all_data(run_dir):
    data = []
    for directory, dirs, fnames in os.walk(run_dir):
        if 'GPU.out' in fnames:
            data.append(parse_logfile(directory + '/GPU.out'))
    return pd.DataFrame(data)

if __name__ == '__main__':
    run_dir = '/users/mscherbela/runs/jaxtest/conv/test10'
    df = read_all_data(run_dir)
    df = df.sort_values(['molecule', 'mcmc_proposal'])
    for i,r in df.iterrows():
        print(f"{r['directory_short']:<40}: {r['error_eval']:+5.1f} +- {1e3*r['sigma_E_eval']:3.1f} mHa ({r['t_opt']:.2f} sec/ep)")









