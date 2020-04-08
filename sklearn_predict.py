from chemprop.args import SklearnPredictArgs
from chemprop.sklearn_predict import predict_sklearn


if __name__ == '__main__':
    predict_sklearn(SklearnPredictArgs().parse_args())
