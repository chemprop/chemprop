.. _predict:

Prediction
----------

To load a trained model and make predictions, run:

.. code-block::
   
    chemprop predict --test-path <test_path> --model-paths <[model_paths]>

where :code:`<test_path>` is the path to the data to test on, and :code:`<[model_paths]>` is the location of checkpoint(s) or model file(s) to use for prediction. It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s). If a directory, will recursively search and predict on all found (.pt) models. By default, predictions will be saved to the same directory as the test path. If desired, a different directory can be specified by using :code:`--preds-path <path>`. The predictions <path> can end with either .csv or .pkl, and the output will be saved to the corresponding file type.

For example:

.. code-block::
  
    chemprop predict --test-path tests/data/smis.csv \
        --model-path tests/data/example_model_v2_regression_mol.ckpt \
        --preds-path preds.csv


Specifying Data to Parse
^^^^^^^^^^^^^^^^^^^^^^^^

By default, Chemprop will assume that the the 0th column in the data .csv will have the data. To use a separate column, specify:

 * :code:`--smiles-columns` Text label of the column that includes the SMILES strings

If atom-mapped reaction SMILES are used, specify:

 * :code:`--reaction-columns` Text labels of the columns that include the reaction SMILES

If :code:`--reaction-mode` was specified during training, those same flags must be specified for the prediction step.

.. _performant-prediction:

Performant Prediction
^^^^^^^^^^^^^^^^^^^^^

Prediction can be accelerated using molecular featurizer package called ``cuik-molmaker``. This package is not installed by default, but can be installed using the script ``check_and_install_cuik_molmaker.py``. In order to enable the accelerated featurizer, use the :code:`--use-cuikmolmaker-featurization` flag. This featurizer also performs on-the-fly featurization of molecules and reduces memory usage which is particularly useful for large datasets.


Uncertainty Quantification
--------------------------

To load a trained model and make uncertainty quantification, run:

.. code-block::
   
    chemprop predict --test-path <test_path> \
        --cal-path <cal_path> \
        --model-paths <[model_paths]> \
        --uncertainty-method <method> \
        --calibration-method <method> \
        --evaluation-methods <[methods]>

where :code:`<test_path>` is the path to the data to test on, :code:`<cal_path>` is the calibration dataset used for uncertainty calibration if needed, and :code:`<[model_paths]>` is the location of checkpoint(s) or model file(s) to use for prediction. The uncertianty estimation, calibration, and evaluations methods are detailed below. 

Uncertainty Estimation
^^^^^^^^^^^^^^^^^^^^^^

The uncertainty of predictions made in Chemprop can be estimated by several different methods. Uncertainty estimation is carried out alongside model value prediction and reported in the predictions csv file when the argument :code:`--uncertainty-method <method>` is provided. If no uncertainty method is provided, then only the model value predictions will be carried out. The available methods are:

 * :code:`dropout`
 * :code:`ensemble`
 * :code:`quantile-regression`
 * :code:`mve`
 * :code:`evidential-total`, :code:`evidential-epistemic`, :code:`evidential-aleatoric`
 * :code:`classification`
 * :code:`classification-dirichlet`
 * :code:`multiclass`
 * :code:`multiclass-dirichlet`

Uncertainty Calibration
^^^^^^^^^^^^^^^^^^^^^^^

Uncertainty predictions may be calibrated to improve their performance on new predictions. Calibration methods are selected using :code:`--calibration-method <method>`, options provided below. An additional dataset to use in calibration is provided through :code:`--cal-path <path>`, along with necessary features like :code:`--cal-descriptors-path <path>`. As with the data used in training, calibration data for multitask models are allowed to have gaps and missing targets in the data.

**Regression**:

 * :code:`zscaling` Assumes that errors are normally distributed according to the estimated variance for each prediction. Applies a constant multiple to all stdev or interval outputs in order to minimize the negative log likelihood for the normal distributions. (https://arxiv.org/abs/1905.11659)
 * :code:`zelikman-interval` Assumes that the error distribution is the same for each prediction but scaled by the uncalibrated standard deviation for each. Multiplies the uncalibrated standard deviation by a factor necessary to cover the specified interval of the calibration set. Does not assume a Gaussian distribution using :code:`--calibration-interval-percentile <float>` which is default ot 95. (https://arxiv.org/abs/2005.12496)
 * :code:`mve-weighting` For use with ensembles of models trained with mve or evidential loss function. Uses a weighted average of the predicted variances to achieve a minimum negative log likelihood of predictions. (https://doi.org/10.1186/s13321-021-00551-x)
 * :code:`conformal-regression` Generates an interval of variable size for each prediction based on quantile predictions of the data such that the actual value has probability :math:`1 - \alpha` of falling in the interval. The desired error rate is controlled using the parameter :code:`--conformal-alpha <float>` which is set by default to 0.1. (https://arxiv.org/abs/2107.07511)

**Classification**:

 * :code:`platt` Uses a linear scaling before the sigmoid function in prediction to minimize the negative log likelihood of the predictions. (https://arxiv.org/abs/1706.04599)
 * :code:`isotonic` Fits an isotonic regression model to the predictions. Prediction outputs are transformed using a stepped histogram-style to match the empirical probability observed in the calibration data. Number and size of the histogram bins are procedurally decided. Histogram bins are wider in the regions of the model output that are less reliable in ordering confidence. (https://arxiv.org/abs/1706.04599)
 * :code:`conformal-multilabel` Generates a pair of sets of labels :math:`C_{in} \subset C_{out}` such that the true set of labels :math:`S` satisfies the property :math:`C_{in} \subset S \subset C_{out}` with probability at least :math:`1-\alpha`. The desired error rate :math:`\alpha` can be controlled with the parameter :code:`--conformal-alpha <float>` which is set by default to 0.1. (https://arxiv.org/abs/2004.10181)


**Multiclass**:

 * :code:`conformal-multiclass` Generates a set of possible classes for each prediction such that the true class has probability :math:`1-\alpha` of falling in the set. The desired error rate :math:`\alpha` can be controlled with the parameter :code:`--conformal-alpha <float>` which is set by default to 0.1. Set generated using the basic conformal method. (https://arxiv.org/abs/2107.07511)
 * :code:`conformal-adaptive` Similar to conformal-multiclass, this method generates a set of possible classes but uses an adaptive conformal method. The desired error rate :math:`\alpha` can be controlled with the parameter :code:`--conformal_alpha <float>` which is set by default to 0.1. (https://arxiv.org/abs/2107.07511)
 * :code:`isotonic-multiclass` Calibrate multiclass classification datasets using isotonic regression. It uses a one-vs-all aggregation scheme to extend isotonic regression from binary to multiclass classifiers. (https://arxiv.org/abs/1706.04599)

Uncertainty Evaluation Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The performance of uncertainty predictions (calibrated or uncalibrated) as evaluated on the test set using different evaluation metrics as specified with :code:`--evaluation-methods <[methods]>`.
Evaluation scores will only appear in the output trace. Multiple evaluation methods can be provided and they will be calculated separately for each model task. Evaluation is only available when the target values are provided with the data in :code:`--test-path <test_path>`. As with the data used in training, evaluation data for multitask models are allowed to have gaps and missing targets in the data.

 .. * Any valid classification or multiclass metric. Because classification and multiclass outputs are inherently probabilistic, any metric used to assess them during training is appropriate to evaluate the confidences produced after calibration.

 * :code:`nll-regression`, :code:`nll-classification`, :code:`nll-multiclass` Returns the average negative log likelihood of the real target as indicated by the uncertainty predictions. Enabled for regression, classification, and multiclass dataset types.
 * :code:`spearman` A regression evaluation metric. Returns the Spearman rank correlation between the predicted uncertainty and the actual error in predictions. Only considers ordering, does not assume a particular probability distribution.
 * :code:`ence` Expected normalized calibration error. A regression evaluation metric. Bins model prediction according to uncertainty prediction and compares the RMSE in each bin versus the expected error based on the predicted uncertainty variance then scaled by variance. (discussed in https://doi.org/10.1021/acs.jcim.9b00975)
 * :code:`miscalibration_area` A regression evaluation metric. Calculates the model's performance of expected probability versus realized probability at different points along the probability distribution. Values range (0, 0.5) with perfect calibration at 0. (discussed in https://doi.org/10.1021/acs.jcim.9b00975)
 * :code:`conformal-coverage-regression`, :code:`conformal-coverage-classification`, :code:`conformal-coverage-multiclass` Measures the empirical coverage of the conformal methods, that is the proportion of datapoints that fall within the output set or interval. Must be used with a conformal calibration method which outputs a set or interval. The metric can be used with multiclass, multilabel, or regression conformal methods.

Different evaluation metrics consider different aspects of uncertainty. It is often appropriate to consider multiple metrics. For intance, miscalibration error is important for evaluating uncertainty magnitude but does not indicate that the uncertainty function discriminates well between different outputs. Similarly, spearman tests ordering but not prediction magnitude.

Evaluations can be used to compare different uncertainty methods and different calibration methods for a given dataset. Using evaluations to compare between datasets may not be a fair comparison and should be done cautiously.