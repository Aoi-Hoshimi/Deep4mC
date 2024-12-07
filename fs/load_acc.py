from fs.AAC import make_acc_vector, make_cc_vector, make_ac_vector
from fs.check_parameters import check_acc_arguments

def ACC_encoding(fastas,method='TAC',type1='DNA',lag = 2):
    all_index = True  # action='store_true', help="Choose all physico-chemical indices, default: False.
    my_property_name, my_property_value, kmer = check_acc_arguments(method, type1, all_index, lag)
    encodings = []
    if method == 'DAC' or method == 'TAC':
        print(method)
        encodings = make_ac_vector(fastas, my_property_name, my_property_value, lag, kmer)
    if method == 'DCC' or method == 'TCC':
        encodings = make_cc_vector(fastas, my_property_name, my_property_value, lag, kmer)
    if method == 'DACC' or method == 'TACC':
        encodings = make_acc_vector(fastas, my_property_name, my_property_value, lag, kmer)
    return encodings


def TAC(fastas,type1='DNA',lag = 2):
    all_index = True
    method = "TAC"
    my_property_name, my_property_value, kmer = check_acc_arguments(method, type1, all_index, lag)
    encodings = make_ac_vector(fastas, my_property_name, my_property_value, lag, kmer)
    return encodings






