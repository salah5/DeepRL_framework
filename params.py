# coding: utf-8
# pytorch version : stable (1.0), mac, conda, python 3.7, None

import warnings

warnings.filterwarnings("ignore")

import argparse

from utils import str2bool

arg_lists = []

parser = argparse.ArgumentParser()


### Methods
##
#
def add_argument_group(name):

    arg = parser.add_argument_group(name)
    
    arg_lists.append(arg)
    
    return arg

###
##
#
def get_config():
    
    config, unparsed = parser.parse_known_args()
    
    return config, unparsed


###
##
#
# # Data
data_arg = add_argument_group('Data')

data_arg.add_argument('--train_path', type=str, default='./data/prep/train.de-en.') # oringal dataset

data_arg.add_argument('--valid_path', type=str, default='./data/prep/valid.de-en.') # oringal dataset

data_arg.add_argument('--test_path', type=str, default='./data/prep/test.de-en.')   # oringal dataset

data_arg.add_argument('--train_x_path', type=str, default='./data/train.de')		# prepaid dataset
	
data_arg.add_argument('--train_y_path', type=str, default='./data/train.en')		# prepaid dataset

data_arg.add_argument('--valid_x_path', type=str, default='./data/valid.de')		# prepaid dataset

data_arg.add_argument('--valid_y_path', type=str, default='./data/valid.en')		# prepaid dataset

data_arg.add_argument('--test_x_path', type=str, default='./data/test.de')			# prepaid dataset

data_arg.add_argument('--test_y_path', type=str, default='./data/test.en')			# prepaid dataset

data_arg.add_argument('--lang_x_pkl', type=str, default='./data/lang.de.pkl')

data_arg.add_argument('--lang_y_pkl', type=str, default='./data/lang.en.pkl')

data_arg.add_argument('--data_dir', type=str, default='./data/')

data_arg.add_argument('--session_dir', type=str, default='./models/')

data_arg.add_argument('--log_dir', type=str, default='./logs/')

# data_arg.add_argument('--load_sess', type=str, default="")

# data_arg.add_argument('--embed_path',type=str, default='../glove.6B/glove.6B.300d.txt')

# #language setting
lang_arg = add_argument_group('lang')

lang_arg.add_argument('-ln','--lang_name', type=list, default=['de','en'])

lang_arg.add_argument('--voc_size',type=list, default=[4000,4000])


# #Preprocessing
prep_arg = add_argument_group('preprocessing')

prep_arg.add_argument('--prep_data', type=str2bool, default= False)



# # Environment
env_arg = add_argument_group('Environment')

env_arg.add_argument('--env_name', type=str, default = 'MT')

env_arg.add_argument('--init_state', type=str, default = 'zero')

env_arg.add_argument('--init_reward', type=str, default = 'zero')

env_arg.add_argument('--delayed_reward', type=str2bool, default = True)

env_arg.add_argument('--embed_size', type=int, default = 4)


env_arg.add_argument('--utt_type', type=str, default='rnn')

env_arg.add_argument('--max_seq_len', type=int, default=25)

env_arg.add_argument('--rnn_cell', type=str, default='gru')

env_arg.add_argument('--enc_cell_size', type=int, default=6) 

env_arg.add_argument('--bi_enc_cell', type=str2bool, default = False)

env_arg.add_argument('--num_layer', type=int, default=1)



env_arg.add_argument('--max_dec_len', type=int, default=25)

env_arg.add_argument('--dec_cell_size', type=int, default=6) # should be the same as enc_cell_size as  it is the init state

env_arg.add_argument('--use_attn', type=str2bool, default=False)

env_arg.add_argument('--use_ptr', type=str2bool, default=False)

env_arg.add_argument('--attn_type', type=str, default='cat')



# # Agent
agent_arg = add_argument_group('Agent')

agent_arg.add_argument('--gen_type', type=str, default='greedy')# greedy, sample, beam, everything != 'beam' is gready, i.e. beam_size=1



# # Network
# net_arg = add_argument_group('Network')
# net_arg.add_argument('--y_size', type=int, default=20)
# net_arg.add_argument('--k', type=int, default=10)
# net_arg.add_argument('--use_mutual', type=str2bool, default=True)
# net_arg.add_argument('--use_reg_kl', type=str2bool, default=True)


# # Training / test parameters
train_arg = add_argument_group('Training')
# train_arg.add_argument('--op', type=str, default='adam')
# train_arg.add_argument('--backward_size', type=int, default=1)
# train_arg.add_argument('--step_size', type=int, default=1)
# train_arg.add_argument('--grad_clip', type=float, default=3.0)
# train_arg.add_argument('--init_w', type=float, default=0.1)
train_arg.add_argument('--init_lr', type=float, default=0.001)
# train_arg.add_argument('--momentum', type=float, default=0.1)
# train_arg.add_argument('--lr_hold', type=int, default=1)
# train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--dropout', type=float, default=0.3)

train_arg.add_argument('--improve_threshold', type=float, default=0.996)

train_arg.add_argument('--patience', type=int, default=10)

train_arg.add_argument('--patient_increase', type=float, default=3.0)

train_arg.add_argument('--early_stop', type=str2bool, default=True)

train_arg.add_argument('--max_epoch', type=int, default=200)


# # MISC
misc_arg = add_argument_group('Misc')

misc_arg.add_argument('--save_model', type=str2bool, default=True)

misc_arg.add_argument('--use_gpu', type=str2bool, default=False)

misc_arg.add_argument('--print_step', type=int, default=2)

misc_arg.add_argument('--ckpt_step', type=int, default=2) # check the model on the valid set 

misc_arg.add_argument('--fix_batch', type=str2bool, default=True) # if batch contains sequences with various length


# misc_arg.add_argument('--include_eod', type=str2bool, default=True)

misc_arg.add_argument('--batch_size', type=int, default=3)

misc_arg.add_argument('--preview_pred', type=str2bool, default=True)

misc_arg.add_argument('--preview_threshold', type=float, default=0.5)

# misc_arg.add_argument('--preview_batch_num', type=int, default=1)

misc_arg.add_argument('--gen_mode', type=str, default='teacher_forcing') # teacher_gen, teacher_forcing

misc_arg.add_argument('--avg_type', type=str, default='word') # for NLL loss

misc_arg.add_argument('--beam_size', type=int, default=1)

# misc_arg.add_argument('--forward_only', type=str2bool, default=False)

misc_arg.add_argument('--seed', type=int, default=1234)

misc_arg.add_argument('--pilot', type=str2bool, default=True)

misc_arg.add_argument('--metric', type=str, default= 'bleu')

