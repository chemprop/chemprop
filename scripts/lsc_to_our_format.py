import os
import shutil

import h5py
import numpy as np
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)


DATASETS = [
    'qm7',
    'qm8',
    'qm9',
    'delaney',
    'freesolv',
    'lipo',
    'pdbbind_full',
    'pdbbind_core',
    'pdbbind_refined',
    'pcba',
    'muv',
    'hiv',
    'bace',
    'bbbp',
    'tox21',
    'toxcast',
    'sider',
    'clintox',
    'chembl'
]


class Args(Tap):
    lsc_dir: str  # Path to directory in lsc save format
    ckpt_dir: str  # Path to directory with targets saved in our format
    save_dir: str  # Path to directory where lsc files will be saved in our format


def lsc_to_our_format(lsc_dir: str, ckpt_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for dataset in DATASETS:
        print(dataset, end='\t')

        success = 0

        # Convert preds and copy over preds and targets
        for fold in range(10):
            lsc_preds_path = os.path.join(lsc_dir, dataset, 'test', f'fold_{fold}', 'semi', 'o0003.evalPredict.hdf5')
            ckpt_targets_path = os.path.join(ckpt_dir, dataset, 'scaffold', str(fold), 'targets.npy')

            if not (os.path.exists(lsc_preds_path) and os.path.exists(ckpt_targets_path)):
                continue

            save_fold_dir = os.path.join(save_dir, dataset, 'scaffold', str(fold))
            os.makedirs(save_fold_dir, exist_ok=True)

            save_preds_path = os.path.join(save_fold_dir, 'preds.npy')
            save_targets_path = os.path.join(save_fold_dir, 'targets.npy')

            # Copy targets
            shutil.copy(ckpt_targets_path, save_targets_path)

            # Convert and copy preds
            preds_file = h5py.File(lsc_preds_path)
            preds = np.array(preds_file['predictions'])
            np.save(save_preds_path, preds)

            success += 1

        print(success)


if __name__ == '__main__':
    args = Args().parse_args()

    lsc_to_our_format(
        lsc_dir=args.lsc_dir,
        ckpt_dir=args.ckpt_dir,
        save_dir=args.save_dir
    )
