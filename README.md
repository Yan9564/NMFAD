# NMFAD
This repository includes code and dataset in paper ["Physics-Enhanced NMF Toward Anomaly Detection in Rotating Mechanical Systems"](https://ieeexplore.ieee.org/document/10579702)
## Data set in experiment
- The data set in the experiment is from literature in ["On the Evaluation of Unsupervised Outlier Detection:
Measures, Datasets, and an Empirical Study"](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)
- The CBCL facial data set used in the appendix is from [http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz](http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz)
- The carriages data set in the case study is available from yanbingxin124@163.com, upon reasonable request.
## Requirements
- Python 3.9.7
- [NIMFA in Python](http://nimfa.biolab.si/): for NMF and variants benchmark methods.
## Code
- Self-defined function
  - GNMF.py: GNMF function
  - PNMF.py: self-defined PNMF function
  - framework_nmf.py : the benchmark NMF method
  - framework_svd.py : the benchmark SVD method
  - framework_svd_st.py : the benchmark SVD method with soft thresholding
  - framework_bd.py : the benchmark BD method
  - framework_bmf.py : the benchmark BMF method
  - framework_snmf.py : the benchmark SNMF method
  - framework_gnmf.py : the benchmark GNMF method
  - framework_lfnmf.py : the benchmark LFNMF method
  - framework_pnmf.py : the proposed PNMF method
- experiments
  - main_experiment_WBC.py : experiment based on the WBC.xlsx dataset, and the results are presented in Figure 5 in the paper.
  - main_face.py : experiment based on the CBCL face dataset, and the results are presented in Figure 15 in the paper.
