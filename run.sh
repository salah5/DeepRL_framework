#!/bin/bash

###
##
# define variables
machine='local' # possible values: {'local', 'gpu'}, where 'local' has no GPU and cuda
 
###
##
# check if we are in RL virtual environment
if [[ $VIRTUAL_ENV == '' ]]
then

    source activate rl
fi

###
##
# export the spec of the running environment (rl)
conda list --explicit > ./documentation/spec-file-$machine.txt 

###
##
# run the latest version of the model by getting it from GitHub
if [[ $machine == 'gpu' ]]
then
	
	git pull
fi



###
##
# Finally execute the model
python run.py 


###
##
# exit from virtual env
source deactivate rl