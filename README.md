# A DeepRL Framework for NLP 

This repository provides a deep reinforcement learning (DeepRL) framework for NLP tasks. It is deep because it utilizes neural network models to represent the concepts in reinforcement learning, e.g., states are encoded by the RNN structure. 

### Note ###

Your interest to this project is highly appreciated. Please give a Github star (on top right) if you use the code in anyhow. 

### Setup ###
* Python 3.7.2
* PyTorch 1.0.1.post2
* More info: [spec-file_*](https://github.com/MMesgar/DeepRL_framework/tree/master/documentation)

### Data ###
For any task, we use a specific dataset. For now we have following tasks and datasets.

* **Machine Translation**

Following [MIXER](https://arxiv.org/pdf/1511.06732.pdf), We use data from the German-English machine translation track of the IWSLT 2014 evaluation campaign. 
The corpus consists of sentence-aligned subtitles of TED and TEDx talks. 


### How to run? ###
* **parameters** 
All parameters are defined in 'params.py'. They all have default values. You may change them either in the file or as arguments of 'run.py' in the 'run.sh' script. 

* **run** 
'''
bash ./run.sh
'''



### Publication ###
