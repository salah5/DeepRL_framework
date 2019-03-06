#!/bin/bash
###
##
# Vairables
comment='update'

if [[ $# > 1 ]]
then
	commnet=$1
fi

###
##
#
git add -u

###
##
# remove files, e.g. params, that you don't want to update
git reset -- params.py

###
##
# Now commit and push
git commit -m$comment

git push