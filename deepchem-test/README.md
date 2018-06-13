# Deepchem Benchmark Test

We tested our model on 12 deepchem benchmark datasets (http://moleculenet.ai/), ranging from physical chemistry to biophysics
properties. To run our model on those datasets,  
```
bash run.sh 1
```
where 1 is the random seed for randomly splitting the dataset into training, validation and testing (not applied to datasets with scaffold splitting).

## Results

We compared our model against the graph convolution in deepchem. Our results are averaged over 3 runs with different random seeds, namely different splits accross datasets.

Results on classification datasets (AUC score, the higher the better)

| Dataset   |	Ours   |	GraphConv (deepchem)   |
| :-------------: |:-------------:| :-----:|
| Bace	| 0.825 ± 0.011	| 0.783 ± 0.014 |
| BBBP	| 0.692 ± 0.015	| 0.690 ± 0.009 |
| Tox21	| 0.849 ± 0.006	| 0.829 ± 0.006 |
| Toxcast	| 0.726 ± 0.014	| 0.716 ± 0.014 |
| Sider |	0.638 ± 0.020	| 0.638 ± 0.012 |
| clintox	| 0.919 ± 0.048	| 0.807 ± 0.047 |
| MUV	| 0.095 ± 0.02 | 0.046 ± 0.031 |
| HIV |	0.763 ± 0.001 |	0.763 ± 0.016 |
| PCBA	| 0.218 ± 0.001 | 	0.136 ± 0.003 | 

Results on regression datasets (Root mean square error, the lower the better)

Dataset	| Ours	| GraphConv/MPNN (deepchem) |
| :-------------: |:-------------:| :-----:|
delaney	| 0.66 ± 0.07 | 	0.58 ± 0.03 |
Freesolv |	1.06 ± 0.19	| 1.15 ± 0.12 |
Lipo |	0.642 ± 0.065 |	0.655 ± 0.036 |
