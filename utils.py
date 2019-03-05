# coding: utf-8
import datetime
import logging

###
##
#
def time():

	localtime = datetime.datetime.now().strftime("%y-%m-%d %H:%M")

	return localtime

###
##
#
def str2bool(v):
    return v.lower() in ('true', '1')

###
##
#
import argparse

def add_args(parser):

	

	return parser

###
##
#
def pars_args():

	parser = argparse.ArgumentParser()

	parser = add_args(parser)

	args, unparsed = parser.parse_known_args()

	return args

###
##
#




	return logger
