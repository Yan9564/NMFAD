# NMFAD
This repository includes code and dataset in paper xxx
## Data set in experiment
- The data set in the experiment is from literature in ["__On the Evaluation of Unsupervised Outlier Detection:
Measures, Datasets, and an Empirical Study__"](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)
- The data set in the case study are available from yanbingxin124@163.com, upon reasonable request.
## Requirements
- Python 3.9.7
- [NIMFA in Python](http://nimfa.biolab.si/): for NMF and variants benchmark methods.
## Code
- Self-defined function
  - GNMF.py: GNMF function
  - PGNMF.py: self-defined PGNMF function
  - framework_nmf.py : the benchmark NMF method
  - framework_svd.py : the benchmark SVD method
  - framework_bd.py : the benchmark BD method
  - framework_bmf.py : the benchmark BMF method
  - framework_snmf.py : the benchmark SNMF method
  - framework_gnmf.py : the benchmark GNMF method
  - framework_lfnmf.py : the benchmark LFNMF method
  - framework_pgnmf.py : the proposed PGNMF method
- experiments
  - main_experiment_WBC.py : experiment based on the WBC.xlsx dataset, and the results are presented in Figure 1 in the paper.
