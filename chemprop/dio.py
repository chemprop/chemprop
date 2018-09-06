import pandas as pd
from sklearn.preprocessing import StandardScaler


def format_list(x,y):
    return [(s[0], [v, ]) for s, v in zip(x, y)]


def load_split_data(data_fn, valid_split, test_split, seed, scale=True):
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(data_fn)
    prop_names = list(set(df.columns) - {"id", "smiles"})
    smiles = df["smiles"].values.reshape(-1, 1)
    y = df[prop_names].values
    if len(y.shape) <2 :
        y = y.reshape(-1, 1)
    split = train_test_split(smiles, y, test_size=test_split, random_state=seed)
    test_x, test_y = split[1], split[3]
    train_x, valid_x, train_y,  valid_y = train_test_split(split[0], split[2], test_size=valid_split)
    if scale:
        _st = StandardScaler()  # zero mean, unit standard deviation
        _st.fit(train_y)
        train_y = _st.transform(train_y)
        valid_y = _st.transform(valid_y)
        test_y = _st.transform(test_y)
    return format_list(train_x, train_y), format_list(valid_x, valid_y), format_list(test_x, test_y)


def get_data(path):
    data = []
    func = lambda x : float(x) if x != '' else None
    with open(path) as f:
        f.readline()
        for line in f:
            vals = line.strip("\r\n ").split(',')
            smiles = vals[0]
            vals = [func(x) for x in vals[1:]]
            data.append((smiles, vals))
    return data

