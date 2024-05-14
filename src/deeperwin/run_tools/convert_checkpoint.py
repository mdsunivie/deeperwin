import pickle
import zipfile
import argparse
import numpy as np
import jax

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_params(params):
    new_params = dict()
    for k, v in params.items():
        if "wf/orbitals/transferable_atomic_orbitals" in k:
            k = k.replace("wf/orbitals/transferable_atomic_orbitals", "wf/~/orbitals/~/transferable_atomic_orbitals")
            new_params[k] = v
        elif k == "wf/~/input/one_el_features/linear_0":
            new_params["wf/~/input/el_ion_features/linear_0"] = v
        elif k == "wf/~/orbitals/~/generalized_atomic_orbitals/backflow_0/mlp/linear_2":
            new_params["wf/~/orbitals/~/generalized_atomic_orbitals/backflow_0/lin_out"] = {'w': v['w']}
        elif k == "wf/~/orbitals/~/generalized_atomic_orbitals/backflow_1/mlp/linear_2":
            new_params["wf/~/orbitals/~/generalized_atomic_orbitals/backflow_1/lin_out"] = {'w': v['w']}
        else:
            new_params[k] = v
    return new_params

def convert_checkpoint(fname_in, fname_out,
                       convert_param_name=False,
                       dump_mcmc_state=False,
                       dump_fixed_params=False,
                       dump_clipping_state=False):
    zf_in = zipfile.ZipFile(fname_in, "r", zipfile.ZIP_BZIP2)
    zf_out = zipfile.ZipFile(fname_out, "a", zipfile.ZIP_BZIP2)
    for item in zf_in.infolist():
        if item.filename == "mcmc_state.pkl" and dump_mcmc_state:
            continue
        if item.filename == "fixed_params.pkl" and dump_fixed_params:
            continue
        if item.filename == "clipping_state.pkl" and dump_clipping_state:
            continue
        if item.filename == "opt_state.pkl":
            continue
        elif item.filename == "params.pkl":
            with zf_in.open(item.filename, "r") as f:
                params = pickle.load(f)
            if convert_param_name:
                params = convert_param(params)
            params = jax.tree_util.tree_map(np.array, params)
            with zf_out.open(item.filename, "w") as f:
                pickle.dump(params, f)
        else:
            buffer = zf_in.read(item.filename)
            zf_out.writestr(item, buffer)
    zf_in.close()
    zf_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, default=".")
    parser.add_argument("--output-file", type=str, required=True, default=".")
    parser.add_argument("--convert-param-name", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--dump-mcmc-state", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--dump-fixed-params", type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--dump-clipping-state", type=str2bool, nargs='?',
                        const=True, default=False)
    args = parser.parse_args()

    convert_checkpoint(args.input_file,
                       args.output_file,
                       args.convert_param_name,
                       args.dump_mcmc_state,
                       args.dump_fixed_params,
                       args.dump_clipping_state)
    print("Finished")

