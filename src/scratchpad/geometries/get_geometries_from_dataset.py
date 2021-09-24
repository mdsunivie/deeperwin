import pandas as pd

def create_changes_string(Rs, E_Refs):
    ret = ''
    for R, E_ref in zip(Rs, E_Refs):
        ret += f'- R: {R}\n  E_ref: {E_ref}\n'
    return ret

if __name__ == '__main__':
    df = pd.read_pickle("Fig2_weight_sharing.pkl.gz")
    #print(pd.unique(df.method))
    df = df[(df.name == 'HChain6') & (df.method == "serial8192ep")]
    for id in df.keys():
        print(id)
    df = df[["physical_ion_positions", "E_ref", "physical_ion_charges"]]
    print(create_changes_string(df.physical_ion_positions,df.E_ref))
