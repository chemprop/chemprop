import os
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--zinc_dir', type=str, required=True,
        help='Path to dir with raw ZINC files')
    parser.add_argument('--max_size', type=int, default=0,
        help='Max number of smiles (0 = all)')
    parser.add_argument('--write_loc', type=str, required=True,
        help='Where to write to')
    parser.add_argument('--individual_files', action='store_true', default=False,
        help='Process files individually')
    args = parser.parse_args()

    if args.individual_files:
        os.makedirs(args.write_loc, exist_ok=True)
        for root, _, names in os.walk(args.zinc_dir):
            for name in tqdm(names, total=len(names)):
                _, ext = os.path.splitext(name)
                if ext == '.txt':
                    with open(os.path.join(root, name), 'r') as rf, \
                            open(os.path.join(args.write_loc, name), 'w') as wf:
                        wf.write('smiles')
                        wf.write('\n')
                        rf.readline()
                        for line in rf:
                            smiles = line.strip().split('\t')[0]
                            wf.write(smiles)
                            wf.write('\n')
    
    else:
        with open(os.path.join(args.write_loc), 'w') as wf:
            wf.write('smiles')
            wf.write('\n')
            count = 0
            for root, _, names in os.walk(args.zinc_dir):
                for name in tqdm(names, total=len(names)):
                    _, ext = os.path.splitext(name)
                    if ext == '.txt':
                        with open(os.path.join(root, name), 'r') as rf:
                            rf.readline()
                            for line in rf:
                                if count > args.max_size:
                                    break
                                smiles = line.strip().split('\t')[0]
                                wf.write(smiles)
                                wf.write('\n')
                                count += 1