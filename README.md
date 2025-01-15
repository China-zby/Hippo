Hippo
----
Hippo allows for high-throughput ingestion within video warehouse. This repository contains a implementation used in the experiments.

Dataset
-------
The dataset consists of these components:

Download and extract the dataset follow the URLs:
https://github.com/favyen/otif

Installation
------------
Clone this repository
```bash
git clone https://anonymous.4open.science/r/Hippo-B74F
```

If using pip, please use the requirements.txt to install all required packages.
Setup conda environment:
```bash
conda create -n hippo python=3.8
conda activate hippo
cd /path/to/Hippo/
pip install -r requirements.txt
```
Install Go:
```bash
sudo apt install golang
```
Run Experiments
---------------
Offline training
---------------
Run Pareto configuration set exploration for each dataset, using both our method and the baselines:
```bash
python ./searchMethod/hippo/eval_configuration_space.py
```
Ingestion Performance
---------------
This section aims to run the ingestion pipeline for each dataset, using both our method and baseline methods for performance testing. To execute the ingestion process, use the following command:
```bash
python ./searchMethod/hippo/parallel_main.py
```
Parameters
------------------
When running the script, you can adjust the following parameters to control the ingestion volume and the methods being used:
- `--test_video_num_list`:100,150,200,250,300
- `--method_list`:hippo,otif,skyscraper,unitune
## Example Commands
Test with 100 video streams, using the default hippo method:
```bash
python ./searchMethod/hippo/parallel_main.py --test_video_num_list 100 --method_list hippo
```
Printing Query Performance 
```bash
python ./pylib/experiments/plot_experiment_one.py
```
Effect of Clustering
------------------
We aim to evaluate the **efficiency** and **performance** of the clustering step in our algorithm. Specifically, we will measure and compare the **execution time** of our algorithm across different video streams.
```bash
python ./searchMethod/hippo/cluster_efficiency.py
```
Ablation Experiment
------------------
Evaluating the contribution of different modules in our algorithm by removing them one at a time and observing the impact on performance.
```bash
python ./searchMethod/hippo/parallel_main.py --test_video_num_list 100 --method_list hippo hippo_without_pareto_reinforcement hippo_without_imitation_learning
```
