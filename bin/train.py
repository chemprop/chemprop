from chemprop.run.parsing import parse_train_args
from chemprop.run.run import cross_validate

if __name__ == '__main__':
    args = parse_train_args()
    cross_validate(args)
