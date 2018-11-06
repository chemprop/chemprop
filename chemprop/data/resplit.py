from argparse import Namespace


def resplit(args: Namespace):
    """
    Resplits the train and validation data chronologically.

    Assumes that the data at train_path and val_path are sorted
    chronologically within each file but have been split randomly
    between the two files. This function puts the first (1 - val_frac)
    of both the train and validation data in the new train_save file
    and puts the remaining val_frac of both in the new val_save file.
    That way, the new validation data comes chronologically after
    the new training data.
    """
    train_frac = 1 - args.val_frac

    # Get train and validation sizes
    with open(args.train_path, 'r') as f:
        train_len = sum(1 for _ in f) - 1
    with open(args.val_path, 'r') as f:
        val_len = sum(1 for _ in f) - 1

    # Resplit data
    with open(args.train_path, 'r') as rtf, open(args.val_path, 'r') as rvf, \
            open(args.train_save, 'w') as wtf, open(args.val_save, 'w') as wvf:
        header = rtf.readline().strip()
        rvf.readline()  # skip header

        wtf.write(header + '\n')
        wvf.write(header + '\n')

        for i in range(train_len):
            line = rtf.readline().strip()

            if i < train_frac * train_len:
                wtf.write(line + '\n')
            else:
                wvf.write(line + '\n')

        for i in range(val_len):
            line = rvf.readline().strip()

            if i < train_frac * val_len:
                wtf.write(line + '\n')
            else:
                wvf.write(line + '\n')
