
# CurriculumText

CurriculumText is an experimental implementation of [FastText](https://fasttext.cc/)
that allows training with curriculum learning. The code implements a few basic 
difficulty measures and training schedulers to boost performance on most text classification 
tasks. 

## Setup

Requirements:

- Install Anaconda
- Run `conda env create -f environment.yml`
- Download the Glove 50B vectors and save & extract them in the `data/` directory


## Run instructions

To reproduce the results, please run the following from the repository root 

```
python -m curriculumtext
```

You will see all experimental results as a long table stored in the `data/results/` directory. 
The jupyter notebook in `notebooks/report.ipynb` reads in the results table and aggregates them
into the format presented in the final report. 

