import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def format_list(x,y):
    return [(s[0], [_v for _v in v]) for s, v in zip(x, y)]


def load_data(data_fn, drop_first=False):
    df = pd.read_csv(data_fn, index_col=0)
    if drop_first:
        df.drop(df.index[0], inplace=True)
    return df

def do_scale(st, y):
    if y.shape[0] > 0:
         return st.transform(y)
    return y


def load_split_data(data_fn, valid_split, test_split, seed, scale=True, drop_first=False, return_index=False):
    df = load_data(data_fn, drop_first)
    prop_names = list(set(df.columns) - {"id", "smiles"})
    smiles = df["smiles"]
    y = df[prop_names]
    if len(y.shape) < 2:
        y = y.reshape(-1, 1)
    split = train_test_split(smiles.values.reshape(-1, 1), y, test_size=test_split, random_state=seed)
    test_x, test_y = split[1], split[3]
    train_x, valid_x, train_y,  valid_y = train_test_split(split[0], split[2], test_size=valid_split, random_state=seed)
    indices = (train_y.index, valid_y.index, test_y.index)  # scaling removed indices, so need to grab here
    if scale:
        _st = StandardScaler()  # zero mean, unit standard deviation
        _st.fit(train_y)
        train_y = do_scale(_st, train_y)
        valid_y = do_scale(_st, valid_y)
        test_y = do_scale(_st, test_y)

    if return_index:
        return format_list(train_x, train_y), format_list(valid_x, valid_y), format_list(test_x, test_y), indices
    else:
        return format_list(train_x, train_y), format_list(valid_x, valid_y), format_list(test_x, test_y)


def get_data(path):
    data = []
    func = lambda x: float(x) if x != '' else None
    with open(path) as f:
        f.readline()
        for line in f:
            vals = line.strip("\r\n ").split(',')
            smiles = vals[0]
            vals = [func(x) for x in vals[1:]]
            data.append((smiles, vals))
    return data

