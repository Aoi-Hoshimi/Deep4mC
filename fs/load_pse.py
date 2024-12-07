#!/usr/bin/env python
# _*_coding:utf-8_*_

import argparse

from fs import check_parameters
from fs.pse import make_PseDNC_vector, make_PseKNC_vector, make_PCPseTNC_vector, make_SCPseDNC_vector, \
    make_SCPseTNC_vector

def pse(fastas,method="PseDNC",kmer=3):
    file = ""
    # method = 'PseDNC' # required=True,choices=['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC'],help="the encoding type")
    lamada =2 # help="The value of lamada; default: 2"
    weight = 0.1 # help="The value of weight; default: 0.1"
    # kmer = 3 # "The value of kmer; it works only with PseKNC method.
    all_index = True # action='store_true', help="Choose all physico-chemical indices, default: False."
    type1='DNA'

    my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(fastas,weight,kmer,lamada,all_index,method,type1)
    encodings = []
    if method == 'PseDNC':
        encodings = make_PseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    if method == 'PseKNC':
        encodings = make_PseKNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight, kmer)
    if method == 'PCPseDNC':  # PseDNC is identical to PC-PseDNC
        encodings = make_PseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    if method == 'PCPseTNC':
        encodings = make_PCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    if method == 'SCPseDNC':
        encodings = make_SCPseDNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    if method == 'SCPseTNC':
        encodings = make_SCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    return encodings

def SCPseTNC(fastas,kmer=3):
    file = ""
    # method = 'PseDNC' # required=True,choices=['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC'],help="the encoding type")
    lamada = 2  # help="The value of lamada; default: 2"
    weight = 0.1  # help="The value of weight; default: 0.1"
    # kmer = 3 # "The value of kmer; it works only with PseKNC method.
    all_index = True  # action='store_true', help="Choose all physico-chemical indices, default: False."
    type1 = 'DNA'
    method = "SCPseTNC"
    my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(fastas,
                                                                                                           weight, kmer,
                                                                                                           lamada,
                                                                                                           all_index,
                                                                                                           method,
                                                                                                           type1)
    encodings = []
    encodings = make_SCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    return encodings

def PCPseTNC(fastas,kmer=3):
    file = ""
    # method = 'PseDNC' # required=True,choices=['PseDNC', 'PseKNC', 'PCPseDNC', 'PCPseTNC', 'SCPseDNC', 'SCPseTNC'],help="the encoding type")
    lamada = 2  # help="The value of lamada; default: 2"
    weight = 0.1  # help="The value of weight; default: 0.1"
    # kmer = 3 # "The value of kmer; it works only with PseKNC method.
    all_index = True  # action='store_true', help="Choose all physico-chemical indices, default: False."
    type1 = 'DNA'
    method = "PCPseTNC"
    my_property_name, my_property_value, lamada_value, weight, kmer = check_parameters.check_Pse_arguments(fastas,
                                                                                                           weight, kmer,
                                                                                                           lamada,
                                                                                                           all_index,
                                                                                                           method,
                                                                                                           type1)
    encodings = []
    encodings = make_PCPseTNC_vector(fastas, my_property_name, my_property_value, lamada_value, weight)
    return encodings