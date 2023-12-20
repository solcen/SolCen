from __future__ import absolute_import, division, print_function
import ctypes
import sys
import logging.config
import numpy as np
from configparser import ConfigParser
from tree_sitter import Language, Parser
from DFG.utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from DFG.DFG import oDFG
import time
import argparse
import glob
import logging
import os
from os import path
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing

cpu_cont = 16


dfg_function = {
    'solidity': oDFG.DFG_solidity
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('./DFG/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser
key_words = ['msg.sender', 'call.value']


def merge(code_tokens, tokens_index):
    for i in range(len(code_tokens)-2):
        if i > len(code_tokens)-3:
            break
        if code_tokens[i]+code_tokens[i+1]+code_tokens[i+2] in key_words:
            #code_tokens[i] = code_tokens[i]+code_tokens[i+1]+code_tokens[i+2]
            #(start0, end0) = tokens_index[i]
            #(start1, end1) = tokens_index[i+2]
            #tokens_index[i] = (start0, end1)
            tokens_index.pop(i+1)
            tokens_index.pop(i+1)
            code_tokens.pop(i+1)
            code_tokens.pop(i+1)

node_list = []
def ergodic(root_node,addr,parent):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        node_list.append([root_node.text.decode(),addr,parent,0])
    else :
        for i in range(0,len(root_node.children)):
            ergodic(root_node.children[i],i,root_node)
def merge_tree(root_node, code):
    ergodic(root_node,0,root_node)
    for i in range(0,len(node_list)-2) :
        if node_list[i][0] + node_list[i+1][0] + node_list[i+2][0] in key_words :
            node_list[i+1][3] = 1
            node_list[i+2][3] = 1
            #parent0 =  node_list[i][2]
            #parent1 =  node_list[i+1][2]
            #parent2 =  node_list[i+2][2]
            #addr0 =  node_list[i][1]
            #addr1 =  node_list[i+1][1]
            #addr2 =  node_list[i+2][1]
            #if parent1 == parent2:
            #    #parent1 = ctypes.cast(id1,type(root_node)).value
            #    del parent1.children[addr1]
            #    del parent1.children[addr1]
            #else :
                #parent1 = ctypes.cast(id1, type(root_node)).value
                #parent2 = ctypes.cast(id2, type(root_node)).value
            #    del parent1.children[addr1]
            #    del parent2.children[addr2]
            #if parent0 == parent1 :
            #    for i in range(0,len(parent0.children)):
            #        parent0.children[i].parent = parent0
            #else :
            #    for i in range(0,len(parent1.children)):
            #        parent1.children[i].parent = parent1
# remove comments, tokenize code and extract dataflow

def sort(x):
    if len(x[4]) == 0:
        return x[1]
    else :
        max_v = max(x[4])
        if max_v > x[1]:
            return max_v+x[1]/max_v
        else :
            return x[1]

def extract_dataflow(code, parser, lang):
    # remove comments
    #try:
        code = remove_comments_and_docstrings(code, lang)
    #except:
    #    pass
    # obtain dataflow
    #if lang == "php":
    #    code = "<?php"+code+"?>"
    #try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]

        #merge(code_tokens, tokens_index)
        #merge_tree(root_node, code)
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        #try:
        cdfg = oDFG()
        #DFG, _ = parser[1](root_node, index_to_code, {},node_list,0)
        DFG, _ = cdfg.DFG_solidity(root_node, index_to_code, {},node_list)
        #except:
        #   DFG = []
        DFG = sorted(DFG, key=lambda x: sort(x))
        # identify critical node in DFG
        critical_idx = []
        for id, e in enumerate(DFG):
            if e[0] == "call" and DFG[id+1][0] == "value":
                critical_idx.append(DFG[id-1][1])
                critical_idx.append(DFG[id+2][1])
        lines = []
        for index, code in index_to_code.items():
            if code[0] in critical_idx:
                line = index[0][0]
                lines.append(line)
        lines = list(set(lines))
        for index, code in index_to_code.items():
            if index[0][0] in lines:
                critical_idx.append(code[0])
        critical_idx = list(set(critical_idx))
        max_nums = 0
        cur_nums = -1
        while cur_nums != max_nums and cur_nums != 0:
            max_nums = len(critical_idx)
            for id, e in enumerate(DFG):
                if e[1] in critical_idx:
                    critical_idx += e[-1]
                for i in e[-1]:
                    if i in critical_idx:
                        critical_idx.append(e[1])
                        break
            critical_idx = list(set(critical_idx))
            cur_nums = len(critical_idx)
        dfg = []
        for id, e in enumerate(DFG):
            if e[1] in critical_idx:
                dfg.append(e)
        dfg = sorted(dfg, key=lambda x: x[1])

        # Removing independent points
        indexs = set()
        for d in dfg:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in dfg:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    #except:
    #    dfg = []
        return code_tokens, DFG


def convert_to_dfg(func):
    time1 = time.perf_counter()
    code_tokens, dfg = extract_dataflow(func, parser, 'solidity')
    time2 = time.perf_counter()
    print(time2 - time1)
    print(code_tokens)
    print(dfg)
    return code_tokens,dfg
#fun = """ {
fun = """function withdraw(uint _amount) public { 
    if(balances[msg.sender] >= _amount) { 
        if(msg.sender.call.value(_amount)()) 
        { _amount = now + 1; } 
        balances[msg.sender] -= now + 1; 
    } 
}"""
fun1 = """function _transfer(address _from, address _to, uint _value) internal { require(_to != 0x0); require(balanceOf[_from] >= _value); require(balanceOf[_to] + _value > balanceOf[_to]); uint previousBalances = balanceOf[_from] + balanceOf[_to]; balanceOf[_from] -= _value; balanceOf[_to] += _value; Transfer(_from, _to, _value); assert(balanceOf[_from] + balanceOf[_to] == previousBalances); }""" 
#convert_to_dfg(fun)
