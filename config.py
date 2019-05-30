"""
Contains settings and parameters for the code benchmarking.py
"""
import os
# input file containing all data including the prediction scores from each method to be benchmarked.
# if a new method is to be added to benchmarking, do epitope predictions for all peptides and add the predicted scores in the last column with header as 'method_name_score', 'method_name' to be replaced by the new method (use alphanumeric characters & underscore only).
# in case the method cannot do prediction for specific peptides (e.g. because of unavailability of model for certain lengths), they should be assigne a bad score e.g.
# if the prediction score for the methods ranges from 0 to 100, 0 being the best predicted epitope, the peptides that cannot be predicted for should be assigned a bad score e.g. 1000

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
input_file_name = ROOT_DIR+'/data/0.complete_data.zip'
result_dir_prefix = ROOT_DIR+'/results/'

# methods to be included in the benchmarking.
# 'all' contains list of all methods
# 'selected' contains some methods selected for specific reasons.
# 'mass_spec_and_netmhcpan_4_l' is for comparing mass-spec & netmhcpan-4-l
# 'test' & 'other' are for testing purposes
# add the new method name in 'all' (add to 'selected' if want to be plotted along with the 'selected' methods as well)
# use only alphanumeric characters and underscore (_)
methods_to_include = {'all': ['pacomplex', 'propredI', 'predep', 'syfpeithi', 'mhclovac', 'bimas', 'rankpep', 'arb', 'smm', 'smmpmbec','consensus', 'netmhc_4', 'netmhcpan_3', 'mhcflurry_B', 'mhcflurry_L', 'netmhcpan_4_B', 'netmhcpan_4_L'],
           'selected': ['syfpeithi','bimas','smmpmbec','consensus','netmhc_4','mhcflurry_B','mhcflurry_L','netmhcpan_4_B','netmhcpan_4_L'],
           'mass_spec_and_netmhcpan_4_l': ['netmhcpan_4_L', 'mass_spec_num_sequenced_by_ms'],
           'test': ['pacomplex', 'bimas','consensus','mhcflurry_B','netmhcpan_4_L'],
           'other': ['smm']}

# this indicates whether the predicted score is directly or inversely prportional to probability of being an epitope or ligand or to binding affinity
# for example in case of SMM, the lower the numerical value of the predicted percentile rank, the better the peptide as a predicted binder. So in this case, the score is inversely proportional to the peptide being an epitope.
score_directions = {'netmhcpan_4': 'inverse', 'netmhcpan_4_B': 'inverse','netmhcpan_4_L': 'inverse', 'netmhcpan_3': 'inverse', # indicates whether the prediction scores is directly/inversely proportional to the response labels (where positive label = 1)
                    'netmhc_4': 'inverse', 'consensus': 'inverse', 'smmpmbec': 'inverse', 'smm': 'inverse', 'arb': 'inverse',
                    'mhcflurry_B': 'inverse', 'mhcflurry_L': 'inverse', 'syfpeithi': 'direct', 'predep': 'inverse', 'rankpep': 'direct',
                    'pacomplex': 'direct', 'bimas': 'direct', 'mhclovac': 'direct', 'propredI': 'direct', 'mass_spec_proteinpilot_confidence': 'direct', 'mass_spec_num_sequenced_by_ms': 'direct', 'mass_spec_combined': 'direct'}

# this indicates the lengths allowed by each method for each allele for prediction
method_lengths_available_dict = {'Db': {'netmhcpan_4_B': [8,9,10,11,12,13], 'netmhcpan_4_L': [8,9,10,11,12,13], 'netmhcpan_3': [8,9,10,11,12,13], 'netmhc_4': [8,9,10,11,12,13], # lengths available for prediction for each method per allele
                                        'consensus': [8,9,10,11,12,13], 'smmpmbec': [8,9,10,11], 'smm': [8,9,10,11], 'arb': [8,9,10,11], 'mhcflurry_B': [8,9,10,11,12,13],
                                        'mhcflurry_L': [8,9,10,11,12,13], 'syfpeithi': [9,10], 'predep': [9], 'rankpep': [8,9,10,11], 'pacomplex': [], 'bimas': [9,10], 'mhclovac': [7,8,9,10,11,12,13], 'propredI': [9]},
                                 'Kb': {'netmhcpan_4_B': [8,9,10,11,12,13], 'netmhcpan_4_L': [8,9,10,11,12,13], 'netmhcpan_3': [8,9,10,11,12,13], 'netmhc_4': [8,9,10,11,12,13],
                                        'consensus': [8,9,10,11,12,13], 'smmpmbec': [8,9,10,11], 'smm': [8,9,10,11], 'arb': [8,9,10,11], 'mhcflurry_B': [8,9,10,11,12,13],
                                        'mhcflurry_L': [8,9,10,11,12,13], 'syfpeithi': [8], 'predep': [8], 'rankpep': [8,9], 'pacomplex': [8], 'bimas': [8,9,10], 'mhclovac': [7,8,9,10,11,12,13], 'propredI': [9]}}

# this indicates the real name of the method with proper format. this is used in the final plots.
method_names_proper = {'mass_spec_proteinpilot_confidence': 'Mass-spec ProteinPilot confidence', 'mass_spec_num_sequenced_by_ms': 'Number of times sequenced by mass-spec', 'mass_spec_combined': 'Mass-spec score',
                       'pacomplex': 'PAComplex', 'propredI': 'ProPredI', 'predep': 'PREDEP', 'syfpeithi': 'SYFPEITHI',
                       'mhclovac': 'MHCLovac', 'bimas': 'BIMAS', 'rankpep': 'Rankpep', 'arb': 'ARB', 'smm': 'SMM',
                       'smmpmbec': 'SMMPMBEC', 'consensus': 'IEDB Consensus', 'netmhc_4': 'NetMHC-4.0', 'netmhcpan_3': 'NetMHCpan-3.0',
                       'mhcflurry_B': 'MHCflurry-B', 'mhcflurry_L': 'MHCflurry_L', 'netmhcpan_4_B': 'NetMHCpan-4.0-B', 'netmhcpan_4_L': 'NetMHCpan-4.0-L'}

# this indicates the colors to be used in curve plots
colors = {'mass_spec_proteinpilot_confidence': 'olive', 'mass_spec_num_sequenced_by_ms': 'orange', 'mass_spec_combined': 'steelblue',
          'pacomplex': 'gray', 'propredI': 'brown', 'predep': 'purple', 'syfpeithi': 'darkkhaki',
          'mhclovac': 'rosybrown', 'bimas': 'royalblue', 'rankpep': 'goldenrod', 'arb': 'violet', 'smm': 'pink',
          'smmpmbec': 'olive', 'consensus': 'orange', 'netmhc_4': 'magenta', 'netmhcpan_3': 'cyan',
          'mhcflurry_B': 'navy', 'mhcflurry_L': 'green', 'netmhcpan_4_B': 'red', 'netmhcpan_4_L': 'black'}

fpr_for_roc_curve_zoom_level = 0.02
max_fpr_for_partial_auc = 0.02
x_percent_values_to_be_captured = [50, 75, 90]
top_x_peptide_counts = [172]