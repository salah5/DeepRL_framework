# A DeepRL Framework for NLP 

This repository provides a deep reinforcement learning (DeepRL) framework for NLP tasks. It is deep because it utilizes neural network models to represent the concepts in reinforcement learning, e.g., states are encoded by the RNN structure. 

### Note ###

Your interest to this project is highly appreciated. Please give a Github star (on top right) if you use the code in anyhow. 

### Setup ###
* Python 3.7.2
* PyTorch 1.0.1.post2
* CUDA Version 9.0.176
* More info: [spec-file_*](https://github.com/MMesgar/DeepRL_framework/tree/master/documentation)

### Data ###
For any task, we use a specific dataset. For now we have following tasks and datasets.

* **Machine Translation**

Following [MIXER](https://arxiv.org/pdf/1511.06732.pdf), We use data from the German-English machine translation track of the IWSLT 2014 evaluation campaign. 
The corpus consists of sentence-aligned subtitles of TED and TEDx talks. 


### Experiments ###
* **parameters** 

All hyper-parameters are defined in 'params.py'. They all have default values. You may change them either in the file or as arguments of 'run.py' in the 'run.sh' script. 
Some of the important parameters for running the project are:
 
 (1) --use_gpu
 (2) --pilot

* **run** 

(1) check/modify the variables in `run.sh`

*  machine: should be 'local' if you run the code on a machine without GPUs, and 'gpu' if you run it on a GPU server
* experiment: is the name of the running experiment. This name is used to save the log file in the logs directory.

(2) run the script in a command line

```
nohup ./run.sh &
```

(3) check the results by looking at the corresponding log file to the experiment in the logs directory

```
tail -f logs/X
```
where X is the name of your experiment


### Publication ###
