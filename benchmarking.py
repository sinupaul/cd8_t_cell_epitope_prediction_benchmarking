"""
Code to run benchmarking of CD8 T cell epitope prediction methods
"""

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import time, datetime
from config import *

print('start')
start_time = time.time()

complete_data = pd.read_csv(input_file_name, compression='zip', sep='\t', low_memory=False) # complete data including peptides, peptide lengths, alleles, labels (+ve/-ve), T-cell response measurements, prediction scores for each method etc.
print('loaded data', u'\u2713', '\n')
total_response_score = complete_data['mean_of_positive_percentage_IFNg_of_CD8'].sum() # get total response in the complete data
total_count = len(complete_data) # get total count of peptide/allele/length combinations
alleles = ['Db', 'Kb'] # alleles involved = H-2-Db & H-2-Kb
complete_data_excluding_minors = pd.DataFrame(complete_data.query('immunogenicity_category != "Minor"'), columns=list(complete_data))
complete_data_excluding_minors_and_modified = pd.DataFrame(complete_data.query('immunogenicity_category != "Minor" and modified_detected=="No"'), columns=list(complete_data))


def make_bar_plot(data_df, sort_item, y_label, bar_color, title, plot_file_name, item):
    """
    Make bar plot from data in data_df
    Args:
        title: title of plot
        sort_item: auc/% peptides
        item: x percent/top peptide
    Returns:
        Generates figures
        2.1.bar_plot_overall_auc_binary_classification_based.png
        3.1.bar_plot_overall_auc_response_based.png
        4.1.amount_of_peptides_to_capture_x_percent_epitopes.png
        5.1.amount_of_peptides_to_capture_x_percent_response.png
        6.1.amount_of_epitopes_captured_in_top_x_peptides.png
        7.1.amount_of_response_captured_in_top_x_peptides.png
    """
    plt.figure(num=None, figsize=(10, 8), dpi=300)
    methods = data_df['method'].tolist()
    plt.grid(True, 'major', 'y', ls='-', lw=.2, color='gray', alpha=.3, zorder=0)
    plt.title(title)
    plt.bar(np.arange(len(methods)), data_df[sort_item].tolist(), width=0.7, color=bar_color, zorder=2)
    plt.ylabel(y_label, fontsize=12)
    if sort_item == '%_peptides':
        if item == 'top_peptide':
            plt.yticks(np.arange(0.0, 1.1, step=0.1), fontsize=12)
            plt.ylim(0,1.05)
            plt.gca().set_yticklabels(["{0:.0f}%".format(x) for x in np.arange(0, 110, step=10)], fontsize=12)
        else:
            plt.yscale('log')
            plt.ylim(0.0001, 1.1)
            plt.gca().set_yticklabels(["{0:.1f}%".format(x * 100) for x in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]], fontsize=12)
    else:
        plt.yticks(np.arange(0.0, 1.1, step=0.1), fontsize=12)
    plt.xlabel('Methods', fontsize=12)
    plt.xticks(np.arange(len(methods)), [method_names_proper[x] for x in data_df['method'].tolist()], rotation=-45, ha='left', rotation_mode="anchor", fontsize=12)
    plt.savefig(result_dir_prefix+plot_file_name, bbox_inches="tight")
    plt.close()
    print('->', plot_file_name, u'\u2713')


def locate_y_coordinate_for_specific_x_coordinate(data_df, zoom_level):
    """
    Identify the y coordinate for the specific x coordinate. This is needed in case of zooming in, to avoid breaking the line before the end of the plot area
    Args:
        zoom_level: The x coordinate upto which the plot needs to be zoomed in
    """
    y2 = np.interp(zoom_level, data_df['x_axis_item'], data_df['y_axis_item'])
    return y2



def locate_y_coordinate_for_specific_x_coordinate_for_x_percent(method, item, score_direction, data_sorted, score_at_x_percent_item, percent_value, type):
    """
    Identify the y coordinate for the specific x coordinate. This is for the functions identifying % peptides to capture epitopes/reponse;
    identify % epitopes/response captured by top x peptides etc.
    Args:
        item: epitope/response
        type: to identify % peptides to capture x% epitopes/response or % epitopes/response in top peptides
    """
    if type=='x_percent':
        if score_direction == 'direct':
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].max()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].min()
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].max()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].min()
        else:
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].min()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_'+item+'_percent'].max()
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].min()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].max()
    else:
        if score_direction == 'direct':
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].max()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].min()
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].max()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].min()
        else:
            y_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].min()
            y_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['cum_sum_' + item + '_percent'].max()
            x_axis_item_plus_1 = data_sorted.query(method + '_score>' + str(score_at_x_percent_item))['count_percent'].min()
            x_axis_item_minus_1 = data_sorted.query(method + '_score<' + str(score_at_x_percent_item))['count_percent'].max()
    if np.isnan(x_axis_item_minus_1):
        x_axis_item_minus_1 = 1.0
    if np.isnan(y_axis_item_minus_1):
        y_axis_item_minus_1 = 1.0
    if np.isnan(x_axis_item_plus_1):
        x_axis_item_plus_1 = 1.0
    if np.isnan(y_axis_item_plus_1):
        y_axis_item_plus_1 = 1.0
    x_axis_item2 = [x_axis_item_minus_1, x_axis_item_plus_1]  # x_axis_item 1 point above & below the needed x_axis_item
    y_axis_item2 = [y_axis_item_minus_1, y_axis_item_plus_1]  # y_axis_item 1 point above & below the needed x_axis_item

    y2 = np.interp(percent_value/100, sorted(x_axis_item2), sorted(y_axis_item2))

    return y2

def make_curves(methods, zoom_or_full, x_axis_type, data_for_curves_dict, title, plot_file_name):
    """
    Make curves - ROC curve from FPR/TPR data, % response vs % peptides etc.
    Args:
        zoom_or_full: shows whether normal full length plot or zoomed in version
        x_axis_type: FPR (for ROC curves) or % peptides (for response based curves)
        title: plot title
    Returns:
        Generates figures
        2.2.roc_curves_binary_classification_based.png
        2.2.roc_curves_binary_classification_based_selected_methods.png
        2.2.roc_curves_binary_classification_based_zoomed_in.png
        2.2.roc_curves_binary_classification_based_zoomed_in_selected_methods.png
        3.2.curves_response_based.png
        8.2.roc_curves_mass_spec_and_netmhcpan_4_l.png
    """
    plt.figure(num=None, figsize=(10, 8), dpi=300)
    zoom_level = 0.02
    if zoom_or_full == 'full':
        plt.plot((0.0, 1.0), (0.0, 1.0), ls="--", c=".3", label='Random (0.500)')
    else:
        plt.plot((0.0, zoom_level), (0.0, zoom_level), ls="--", c=".3", label='Random (0.500)')
    for i in range(len(methods)):
        method = methods[i]
        color = colors[method]
        x_axis_item = data_for_curves_dict[method]['x_axis_item']
        y_axis_item = data_for_curves_dict[method]['y_axis_item']
        result_auc = data_for_curves_dict[method]['result_auc']
        count_of_x_axis_items_at_zoom_level = len([x for x in x_axis_item if x <= zoom_level])
        x_y_df = pd.DataFrame(data={'x_axis_item': x_axis_item, 'y_axis_item': y_axis_item})
        if zoom_or_full == 'full':
            plt.plot(x_axis_item, y_axis_item, color=color, label='%s (%0.3f)' % (method_names_proper[method], result_auc))
            plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
        else: # here it's zoomed in version
            y_coordinate_for_zoom_level = locate_y_coordinate_for_specific_x_coordinate(x_y_df, zoom_level)
            x_axis_item_new = x_axis_item[:count_of_x_axis_items_at_zoom_level]
            y_axis_item_new = y_axis_item[:count_of_x_axis_items_at_zoom_level]
            x_axis_item_new.append(zoom_level)
            y_axis_item_new.append(y_coordinate_for_zoom_level)
            plt.plot(x_axis_item_new, y_axis_item_new, color=color, label='%s (%0.3f)' % (method_names_proper[method], result_auc))
        if x_axis_type == 'fpr':
            plt.xlabel('False positive rate', fontsize=12)
            plt.ylabel('True positive rate', fontsize=12)
            plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
        else: # here it's % peptides
            plt.xlabel('% peptides', fontsize=12)
            plt.ylabel('% response', fontsize=12)
            plt.gca().set_yticklabels(["{0:.0f}%".format(x*100) for x in np.arange(0.0, 1.1, 0.1)], fontsize=12)
            plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
    plt.ylim(-0.05, 1.05)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title='Methods (AUC)', loc='lower right')
    plt.title(title)
    plt.savefig(result_dir_prefix+plot_file_name)
    plt.close()
    print('->', plot_file_name, u'\u2713')


def prediction_score_reformat(method, data_df, score_direction, score_name):
    """
    Reformatting the prediction scores based on the relation of the prediction score & positive labels, i.e. directly proportional or inversely proportional.
    This is based on score_directions dict
    Args:
        score_direction: score direction of the method - inverse/direct
        score_name: the name of the score for each method, which is basically method name + '_score'
    Returns:
        1. data with re-formatted prediction scores
    """
    if score_direction == 'inverse':
        data_df[score_name + '_updated'] = data_df[score_name].apply(lambda x: 1 / (x + 0.00001))
        if method == 'predep':
            data_df[score_name + '_updated'] = data_df[score_name].apply(lambda x: 100 - x)
    else:
        data_df[score_name + '_updated'] = data_df[score_name]
    return data_df[score_name + '_updated']


def lengthwise_auc_binary_classification_based(data, methods):
    """
    Calculate AUCs from ROC (TPR vs. FPR) for each method on a per length/allele basis. AUCs are derived from binary classification of peptides as epitope/non-epitope.
    Args:
        data: the complete data of prediction scores, labels & response values
    Returns:
        Generates file
        1.lengthwise_auc_binary_classification_based.tsv - AUC values for all methods - lengthwise AUCs, average of lengthwise AUCs per method/allele and average of allelewise AUCs.
    """

    print('lengthwise auc - binary classification based - start')
    l = 0 # for indexing & populating the df
    lengthwise_auc_df = pd.DataFrame(columns=['method', 'allele', 'length', 'auc'])

    for i in range(len(methods)):
        method = methods[i]
        for j in range(len(alleles)):
            allele = alleles[j]
            lengths = method_lengths_available_dict[allele][method] # getting the available lengths for this method from the length dict
            for k in range(len(lengths)):
                length = lengths[k]
                query = 'allele == "'+allele+'" & length == '+str(length) # getting subgroup
                complete_data_excluding_minors_per_allele_per_length = pd.DataFrame(data.query(query), columns=list(data))
                if complete_data_excluding_minors_per_allele_per_length['label'].sum() == 0: # to account for allele/length categories where there are no +ves
                    result_auc = float('nan')
                else:
                    score_name = method + '_score'
                    score_direction = score_directions[method]
                    complete_data_excluding_minors_per_allele_per_length[score_name + '_updated'] = prediction_score_reformat(method, complete_data_excluding_minors_per_allele_per_length, score_direction, score_name)
                    labels = complete_data_excluding_minors_per_allele_per_length['label'].tolist()
                    predicted_scores = complete_data_excluding_minors_per_allele_per_length[score_name + '_updated'].tolist()

                    fpr, tpr, thresholds = metrics.roc_curve(labels, predicted_scores)
                    result_auc = metrics.auc(fpr, tpr)

                lengthwise_auc_df.loc[l] = (method, allele, length, result_auc)
                l+=1
        print(str(i+1)+'.'+method, u'\u2713')

    lengthwise_auc_df_grouped_method_allele = lengthwise_auc_df.groupby(['method', 'allele']) # for calculating average AUC per method per allele
    for method_allele_group in lengthwise_auc_df_grouped_method_allele:
        lengthwise_auc_df.loc[l] = (method_allele_group[0][0], method_allele_group[0][1], 'average', method_allele_group[1]['auc'].mean())
        l+=1

    lengthwise_average_auc_df = lengthwise_auc_df.query('length=="average"') # for calculating average AUC per method
    lengthwise_auc_df_grouped_method = lengthwise_average_auc_df.groupby(['method'])
    for method_group in lengthwise_auc_df_grouped_method:
        lengthwise_auc_df.loc[l] = (method_group[0], 'average', '-', method_group[1]['auc'].mean())
        l+=1

    lengthwise_auc_df = lengthwise_auc_df.replace({pd.np.nan: '*'})
    lengthwise_auc_df.loc[l] = ('*No positives & hence AUC cannot be estimated for the allele/length class','','','')
    lengthwise_auc_df.to_csv(result_dir_prefix+'1.lengthwise_auc_binary_classification_based.tsv', sep='\t', index=False)
    print('->', '1.lengthwise_auc_binary_classification_based.tsv', u'\u2713')
    print('lengthwise auc - binary classification based - done', u'\u2713', '\n')


def overall_auc_binary_classification_based(data, methods, max_fpr_for_partial_auc):
    """
    Calculate overall AUC from ROC (TPR vs. FPR). "Overall" means all peptides of all lengths and alleles taken together (not allelewise or lengthwise). AUCs are derived from binary classification of peptides as epitope/non-epitope.
    Args:
        data: the complete data of prediction scores, labels & response values
    Returns:
        Generates file
        2.overall_auc_binary_classification_based.tsv - AUC values for all methods - binary classification based
    """

    print('overall auc - binary classification based - start')
    data_for_curves_dict = {}
    overall_auc_binary_classification_based_df = pd.DataFrame(columns=['method', 'auc'])
    overall_auc_binary_classification_based_df_partial_auc = pd.DataFrame(columns=['method', 'partial_auc'])
    labels = data['label'].tolist()

    for i in range(len(methods)):
        method = methods[i]
        data_for_curves_dict[method] = {}
        score_name = method+'_score'
        score_direction = score_directions[method]
        data[score_name + '_updated'] = prediction_score_reformat(method, data, score_direction, score_name)
        predicted_scores = data[score_name + '_updated'].tolist()

        fpr, tpr, thresholds = metrics.roc_curve(labels, predicted_scores)
        result_auc = metrics.auc(fpr, tpr)
        result_auc_score_partial = metrics.roc_auc_score(labels, predicted_scores, max_fpr=max_fpr_for_partial_auc)

        overall_auc_binary_classification_based_df.loc[i] = (method, result_auc)
        overall_auc_binary_classification_based_df_partial_auc.loc[i] = (method, result_auc_score_partial)
        data_for_curves_dict[method]['x_axis_item'] = fpr.tolist()
        data_for_curves_dict[method]['y_axis_item'] = tpr.tolist()
        data_for_curves_dict[method]['result_auc'] = result_auc
        print(str(i+1)+'.'+method, u'\u2713')

    if len(methods) > 10 and len(data['modified_detected'].drop_duplicates()) != 1:
        overall_auc_binary_classification_based_df_sorted = overall_auc_binary_classification_based_df.sort_values(by=['auc'], ascending=False)
        overall_auc_binary_classification_based_df_sorted.to_csv(result_dir_prefix+'2.overall_auc_binary_classification_based.tsv', sep='\t', index=False)
        print('->', '2.overall_auc_binary_classification_based.tsv', u'\u2713')
        overall_auc_binary_classification_based_df_partial_auc_sorted = overall_auc_binary_classification_based_df_partial_auc.sort_values(by=['partial_auc'], ascending=False)
        overall_auc_binary_classification_based_df_partial_auc_sorted = overall_auc_binary_classification_based_df_partial_auc_sorted.append({'method': '*partial auc at fpr = ' + str(max_fpr_for_partial_auc)}, ignore_index=True)
        overall_auc_binary_classification_based_df_partial_auc_sorted.to_csv(result_dir_prefix+'2.overall_auc_binary_classification_based_partial_auc.tsv', sep='\t', index=False)
        print('->', '2.overall_auc_binary_classification_based_partial_auc.tsv', u'\u2713')
        make_bar_plot(overall_auc_binary_classification_based_df_sorted, 'auc', 'AUC', 'steelblue', 'Overall AUC (epitope binary classification based)', '2.1.bar_plot_overall_auc_binary_classification_based.png', 'overall_auc')
        make_curves(methods, 'full', 'fpr', data_for_curves_dict, 'ROC curves (epitope binary classification based; all methods)','2.2.roc_curves_binary_classification_based.png')
        make_curves(methods, 'zoom', 'fpr', data_for_curves_dict, 'ROC curves (epitope binary classification based; all methods; zoomed-in to FPR = 0.02)','2.2.roc_curves_binary_classification_based_zoomed_in.png')
    elif len(methods) > 10 and len(data['modified_detected'].drop_duplicates()) == 1:
        overall_auc_binary_classification_based_df_sorted = overall_auc_binary_classification_based_df.sort_values(by=['auc'], ascending=False)
        overall_auc_binary_classification_based_df_sorted.to_csv(result_dir_prefix+'9.overall_auc_binary_classification_based_excluding_modified_peptides.tsv', sep='\t', index=False)
        print('->', '9.overall_auc_binary_classification_based_excluding_modified_peptides.tsv', u'\u2713')
    elif len(methods) > 2:
        make_curves(methods, 'full', 'fpr', data_for_curves_dict, 'ROC curves (epitope binary classification based; selected methods)','2.2.roc_curves_binary_classification_based_selected_methods.png')
        make_curves(methods, 'zoom', 'fpr', data_for_curves_dict, 'ROC curves (epitope binary classification based; selected methods; zoomed-in to FPR = 0.02)','2.2.roc_curves_binary_classification_based_zoomed_in_selected_methods.png')
    else:
        make_curves(methods, 'full', 'fpr', data_for_curves_dict, 'ROC curves (based on Mass-spectrometry and NetMHCpan-4-L)','8.2.roc_curves_mass_spec_and_netmhcpan_4_l.png')
        make_curves(methods, 'zoom', 'fpr', data_for_curves_dict, 'ROC curves (based on Mass-spectrometry and NetMHCpan-4-L; zoomed-in to FPR = 0.02)', '8.2.roc_curves_mass_spec_and_netmhcpan_4_l_zoomed_in.png')
    print('overall auc - binary classification based - done', u'\u2713', '\n')


def overall_auc_response_based(data, methods):
    """
    Calculate overall AUC derived from the T-cell responses of the peptides. AUC is calculated based on T-cell response vs. % peptides
    Note: Here "minors" are not excluded. Response in all peptides are taken into account
    Args:
        data: the complete data of prediction scores, labels & response values
    Returns:
        Generates file
        3.overall_auc_response_based.tsv - AUC values for all methods - response based
    """

    print('overall auc - response based - start')
    data_for_curves_dict = {}
    overall_auc_response_based_df = pd.DataFrame(columns=['method', 'auc'])

    for i in range(len(methods)):
        method = methods[i]
        data_for_curves_dict[method] = {}
        score_name = method+'_score'
        score_direction = score_directions[method]

        if score_direction == 'inverse':
            data_sorted = data.sort_values(by=[score_name], ascending=True)
        else:
            data_sorted = data.sort_values(by=[score_name], ascending=False)

        data_sorted['peptides_cumu_count'] = np.arange(1, total_count+1)
        data_sorted['peptides_count_percent'] = data_sorted['peptides_cumu_count'].apply(lambda x: x/total_count)

        data_sorted['response_score_cumu_sum'] = data_sorted['mean_of_positive_percentage_IFNg_of_CD8'].cumsum()
        data_sorted['response_score_percent'] = data_sorted['response_score_cumu_sum'].apply(lambda x: x/total_response_score)

        data_sorted['score_change'] = data_sorted[score_name]==data_sorted[score_name].shift(-1)
        data_thresholds = data_sorted.query('score_change==False')

        peptides_count_percent = [0.0]+data_thresholds['peptides_count_percent'].tolist()
        response_score_percent = [0.0]+data_thresholds['response_score_percent'].tolist()

        result_auc = metrics.auc(peptides_count_percent, response_score_percent)
        overall_auc_response_based_df.loc[i] = (method, result_auc)

        data_for_curves_dict[method]['x_axis_item'] = peptides_count_percent
        data_for_curves_dict[method]['y_axis_item'] = response_score_percent
        data_for_curves_dict[method]['result_auc'] = result_auc
        print(str(i+1)+'.'+method, u'\u2713')

    overall_auc_response_based_df_sorted = overall_auc_response_based_df.sort_values(by=['auc'], ascending=False)
    methods = overall_auc_response_based_df_sorted['method'].tolist()
    methods.reverse()
    overall_auc_response_based_df_sorted.to_csv(result_dir_prefix+'3.overall_auc_response_based.tsv', sep='\t', index=False)
    print('->', '3.overall_auc_response_based.tsv', u'\u2713')
    make_bar_plot(overall_auc_response_based_df_sorted, 'auc', 'AUC', 'indianred', 'Overall AUC (T-cell response based)', '3.1.bar_plot_overall_auc_response_based.png', 'overall_auc')
    make_curves(methods, 'full', 'peptide_percent', data_for_curves_dict, '% T-cell response vs. % peptides (all methods)', '3.2.curves_response_based.png')
    make_curves(methods, 'zoom', 'peptide_percent', data_for_curves_dict, '% T-cell response vs. % peptides (all methods; zoomed-in to FPR = 0.02)', '3.2.curves_response_based_zoomed_in.png') # ==> not making zoomed in curve for the response based. This is because since the thresholds are sampled, the zoomed in version may look highly distorted which may be misleading.
    print('overall auc - response based - done', u'\u2713', '\n')


def peptides_to_capture_x_percent_epitope(data, methods, percent_value):
    """
    Calculates the amount of peptides needed to capture x% epitopes
    Args:
         percent_value: the percent value - need to be given as integer e.g. 50 (not 50% or 0.5)
    Returns:
        Generates file
        4.amount_of_peptides_to_capture_x_percent_epitopes
    """
    print('peptides needed to capture ' +str(percent_value)+ '% epitopes - start')
    total_count_excluding_minors = len(data) # total peptides
    total_positives = data['label'].sum() # total epitopes
    amount_to_capture = round(total_positives*percent_value/100) # % amount needed e.g. 42 if 50% with total 83 epitopes
    percent_peptides_needed = []
    percent_peptides_needed_df = pd.DataFrame()
    for i in range(len(methods)):
        method = methods[i]
        score_name = method+'_score'
        score_direction = score_directions[method]
        if score_direction == 'inverse':
            data_sorted = data.sort_values(by=[score_name], ascending=True)
        else:
            data_sorted = data.sort_values(by=[score_name], ascending=False)
        data_sorted['cum_sum_epitope'] = data_sorted['label'].cumsum()
        data_sorted['cum_sum_epitope_percent'] = data_sorted['cum_sum_epitope'].apply(lambda x: x/total_positives)
        data_sorted['count'] = [x+1 for x in np.arange(total_count_excluding_minors)]
        data_sorted['count_percent'] = data_sorted['count'].apply(lambda x: x/(total_count_excluding_minors))
        if score_direction == 'direct':
            score_at_x_percent_item = data_sorted.loc[data_sorted['cum_sum_epitope'] == amount_to_capture][score_name].max() # identifies the score at 50% epitopes e.g. 21 for syf
        else:
            score_at_x_percent_item = data_sorted.loc[data_sorted['cum_sum_epitope'] == amount_to_capture][score_name].min() # identifies the score at 50% epitopes e.g. 21 for syf
        count_percent_needed = locate_y_coordinate_for_specific_x_coordinate_for_x_percent(method, 'epitope', score_direction, data_sorted, score_at_x_percent_item, percent_value, 'x_percent')
        percent_peptides_needed.append(count_percent_needed)
        print(str(i + 1) + '.' + method, u'\u2713')

    percent_peptides_needed_df['method'] = methods
    percent_peptides_needed_df['%_peptides'] = percent_peptides_needed
    percent_peptides_needed_df['count_peptides'] = percent_peptides_needed_df['%_peptides'].apply(lambda x: round(x*total_count_excluding_minors))
    percent_peptides_needed_df_sorted = percent_peptides_needed_df.sort_values(by=['%_peptides'], ascending=True)
    make_bar_plot(percent_peptides_needed_df_sorted, '%_peptides', '% peptides (log scale)', 'steelblue', '% peptides needed to capture '+str(percent_value)+ '% epitopes', '4.1.amount_of_peptides_to_capture_'+str(percent_value)+'_percent_epitopes.png', 'x_percent')
    percent_peptides_needed_df_sorted['%_peptides'] = ["{0:.4f}%".format(x * 100) for x in percent_peptides_needed_df_sorted['%_peptides']]
    percent_peptides_needed_df_sorted.to_csv(result_dir_prefix+'4.amount_of_peptides_to_capture_'+str(percent_value)+'_percent_epitopes.tsv', sep='\t', index=False)
    print('->', '4.amount_of_peptides_to_capture_'+str(percent_value)+'_percent_epitopes.tsv', u'\u2713')
    print('peptides needed to capture ' +str(percent_value)+ '% epitopes - done', u'\u2713', '\n')


def peptides_to_capture_x_percent_response(data, methods, percent_value):
    """
    Calculates the amount of peptides needed to capture x% T-cell response
    Args:
         percent_value: the percent value - need to be given as integer e.g. 50 (not 50% or 0.5)
    Returns:
        Generates file
        5.amount_of_peptides_to_capture_x_percent_response
    """
    print('peptides needed to capture ' +str(percent_value)+ '% response - start')
    amount_to_capture = total_response_score*percent_value/100
    percent_peptides_needed = []
    percent_peptides_needed_df = pd.DataFrame()
    for i in range(len(methods)):
        method = methods[i]
        score_name = method+'_score'
        score_direction = score_directions[method]
        if score_direction == 'inverse':
            data_sorted = data.sort_values(by=[score_name], ascending=True)
        else:
            data_sorted = data.sort_values(by=[score_name], ascending=False)
        data_sorted['cum_sum_response'] = data_sorted['mean_of_positive_percentage_IFNg_of_CD8'].cumsum()
        data_sorted['cum_sum_response_percent'] = data_sorted['cum_sum_response'].apply(lambda x: x/total_response_score)
        data_sorted['count'] = [x+1 for x in np.arange(total_count)]
        data_sorted['count_percent'] = data_sorted['count'].apply(lambda x: x/(total_count))
        if score_direction == 'direct':
            score_at_x_percent_item = data_sorted.loc[data_sorted['cum_sum_response'] >= amount_to_capture][score_name].max() # identifies the score at 50% response e.g. 21 for syf
        else:
            score_at_x_percent_item = data_sorted.loc[data_sorted['cum_sum_response'] >= amount_to_capture][score_name].min() # identifies the score at 50% response e.g. 21 for syf
        count_percent_needed = locate_y_coordinate_for_specific_x_coordinate_for_x_percent(method, 'response', score_direction, data_sorted, score_at_x_percent_item, percent_value, 'x_percent')
        percent_peptides_needed.append(count_percent_needed)
        print(str(i + 1) + '.' + method, u'\u2713')

    percent_peptides_needed_df['method'] = methods
    percent_peptides_needed_df['%_peptides'] = percent_peptides_needed
    percent_peptides_needed_df['count_peptides'] = percent_peptides_needed_df['%_peptides'].apply(lambda x: round(x*total_count))
    percent_peptides_needed_df_sorted = percent_peptides_needed_df.sort_values(by=['%_peptides'], ascending=True)
    make_bar_plot(percent_peptides_needed_df_sorted, '%_peptides', '% peptides (log scale)', 'indianred', '% peptides needed to capture '+str(percent_value)+ '% response', '5.1.amount_of_peptides_to_capture_'+str(percent_value)+'_percent_response.png', 'x_percent')
    percent_peptides_needed_df_sorted['%_peptides'] = ["{0:.4f}%".format(x * 100) for x in percent_peptides_needed_df_sorted['%_peptides']]
    percent_peptides_needed_df_sorted.to_csv(result_dir_prefix+'5.amount_of_peptides_to_capture_'+str(percent_value)+'_percent_response.tsv', sep='\t', index=False)
    print('->', '5.amount_of_peptides_to_capture_'+str(percent_value)+'_percent_response.tsv', u'\u2713')
    print('peptides needed to capture ' +str(percent_value)+ '% response - done', u'\u2713', '\n')


def how_much_epitopes_captured_in_top_peptides(data, methods, top_peptide_count):
    """
    Calculates the amount of epitopes captured in top x peptides. 172 is default because that's the number of peptides identified by LC-MS/MS
    Args:
         top_peptide_count: how many top peptides need to be considered
    Returns:
        Generates file
        6.amount_of_epitopes_captured_in_top_peptides.tsv
    """
    print('how much epitopes captured in top ' +str(top_peptide_count)+ ' peptides - start')
    total_positives = data['label'].sum()
    total_count_excluding_minors = len(data)
    percent_value = (top_peptide_count/total_count_excluding_minors)*100
    percent_epitopes_collected = []
    percent_epitopes_collected_df = pd.DataFrame()
    for i in range(len(methods)):
        method = methods[i]
        score_name = method+'_score'
        score_direction = score_directions[method]
        if score_direction == 'inverse':
            data_sorted = data.sort_values(by=[score_name], ascending=True)
        else:
            data_sorted = data.sort_values(by=[score_name], ascending=False)
        data_sorted['cum_sum_epitope'] = data_sorted['label'].cumsum()
        data_sorted['cum_sum_epitope_percent'] = data_sorted['cum_sum_epitope'].apply(lambda x: x/total_positives)
        data_sorted['count'] = [x+1 for x in np.arange(total_count_excluding_minors)]
        data_sorted['count_percent'] = data_sorted['count'].apply(lambda x: x/(total_count_excluding_minors))
        score_at_top_peptide_count = data_sorted.iloc[top_peptide_count-1][score_name]
        percent_epitopes_captured = locate_y_coordinate_for_specific_x_coordinate_for_x_percent(method, 'epitope', score_direction, data_sorted, score_at_top_peptide_count, percent_value, 'top_peptide')
        percent_epitopes_collected.append(percent_epitopes_captured)
        print(str(i + 1) + '.' + method, u'\u2713')

    percent_epitopes_collected_df['method'] = methods
    percent_epitopes_collected_df['%_peptides'] = percent_epitopes_collected
    percent_epitopes_collected_df['count_epitopes'] = percent_epitopes_collected_df['%_peptides'].apply(lambda x: round(x*total_positives))
    percent_epitopes_collected_df_sorted = percent_epitopes_collected_df.sort_values(by=['%_peptides'], ascending=False)
    make_bar_plot(percent_epitopes_collected_df_sorted[['method', '%_peptides']], '%_peptides', '% epitopes', 'steelblue', '% epitopes captured in top ' +str(top_peptide_count)+ ' peptides', '6.1.amount_of_epitopes_captured_in_top_'+str(top_peptide_count)+'_peptides.png', 'top_peptide')
    percent_epitopes_collected_df_sorted['%_peptides'] = ["{0:.4f}%".format(x * 100) for x in percent_epitopes_collected_df_sorted['%_peptides']]
    percent_epitopes_collected_df_sorted.rename(index=str, columns={"%_peptides": "%_epitopes"}, inplace=True)
    percent_epitopes_collected_df_sorted.to_csv(result_dir_prefix+'6.amount_of_epitopes_captured_in_top_'+str(top_peptide_count)+'_peptides.tsv', sep='\t', index=False)
    print('->', '6.amount_of_epitopes_captured_in_top_'+str(top_peptide_count)+'_peptides.tsv', u'\u2713')
    print('how much epitopes captured in top ' +str(top_peptide_count)+ ' peptides - done', u'\u2713', '\n')


def how_much_response_captured_in_top_peptides(data, methods, top_peptide_count):
    """
    Calculates the amount of response captured in top x peptides. 172 is default because that's the number of peptides identified by LC-MS/MS
    Args:

    Returns:
        Generates file
        7.amount_of_response_captured_in_top_peptides.tsv
    """
    print('how much response captured in top ' +str(top_peptide_count)+ ' peptides - start')
    percent_value = (top_peptide_count/total_count)*100
    percent_response_collected = []
    percent_response_collected_df = pd.DataFrame()
    for i in range(len(methods)):
        method = methods[i]
        score_name = method+'_score'
        score_direction = score_directions[method]
        if score_direction == 'inverse':
            data_sorted = data.sort_values(by=[score_name], ascending=True)
        else:
            data_sorted = data.sort_values(by=[score_name], ascending=False)
        data_sorted['cum_sum_response'] = data_sorted['mean_of_positive_percentage_IFNg_of_CD8'].cumsum()
        data_sorted['cum_sum_response_percent'] = data_sorted['cum_sum_response'].apply(lambda x: x/total_response_score)
        data_sorted['count'] = [x+1 for x in np.arange(total_count)]
        data_sorted['count_percent'] = data_sorted['count'].apply(lambda x: x/(total_count))
        score_at_top_peptide_count = data_sorted.iloc[top_peptide_count-1][score_name]
        percent_response_captured = locate_y_coordinate_for_specific_x_coordinate_for_x_percent(method, 'response', score_direction, data_sorted, score_at_top_peptide_count, percent_value, 'top_peptide')
        percent_response_collected.append(percent_response_captured)
        print(str(i + 1) + '.' + method, u'\u2713')

    percent_response_collected_df['method'] = methods
    percent_response_collected_df['%_peptides'] = percent_response_collected
    percent_response_collected_df['count_response'] = percent_response_collected_df['%_peptides'].apply(lambda x: round(x*total_response_score,2))
    percent_response_collected_df_sorted = percent_response_collected_df.sort_values(by=['%_peptides'], ascending=False)
    make_bar_plot(percent_response_collected_df_sorted[['method', '%_peptides']], '%_peptides', '% response', 'indianred', '% T-cell response captured in top ' +str(top_peptide_count)+ ' peptides', '7.1.amount_of_response_captured_in_top_'+str(top_peptide_count)+'_peptides.png', 'top_peptide')
    percent_response_collected_df_sorted['%_peptides'] = ["{0:.4f}%".format(x * 100) for x in percent_response_collected_df_sorted['%_peptides']]
    percent_response_collected_df_sorted.rename(index=str, columns={"%_peptides": "%_response"}, inplace=True)
    percent_response_collected_df_sorted.to_csv(result_dir_prefix+'7.amount_of_response_captured_in_top_'+str(top_peptide_count)+'_peptides.tsv', sep='\t', index=False)
    print('->', '7.amount_of_response_captured_in_top_'+str(top_peptide_count)+'_peptides.tsv', u'\u2713')
    print('how much response captured in top ' +str(top_peptide_count)+ ' peptides - done', u'\u2713', '\n')


methods_all = methods_to_include['all']
methods_selected = methods_to_include['selected']
methods_mass_spec_and_netmhcpan_4_l = methods_to_include['mass_spec_and_netmhcpan_4_l']
methods_test = methods_to_include['test']
methods_other = methods_to_include['other']

lengthwise_auc_binary_classification_based(complete_data_excluding_minors, methods_all)
overall_auc_binary_classification_based(complete_data_excluding_minors, methods_all, max_fpr_for_partial_auc)
overall_auc_binary_classification_based(complete_data_excluding_minors, methods_selected, max_fpr_for_partial_auc)
overall_auc_response_based(complete_data, methods_all)
for i in range(len(x_percent_values_to_be_captured)):
    percent_value = x_percent_values_to_be_captured[i]
    peptides_to_capture_x_percent_epitope(complete_data_excluding_minors, methods_all, percent_value)
for i in range(len(x_percent_values_to_be_captured)):
    percent_value = x_percent_values_to_be_captured[i]
    peptides_to_capture_x_percent_response(complete_data, methods_all, percent_value)
for i in range(len(top_x_peptide_counts)):
    top_peptide_count = top_x_peptide_counts[i]
    how_much_epitopes_captured_in_top_peptides(complete_data_excluding_minors, methods_all, top_peptide_count)
for i in range(len(top_x_peptide_counts)):
    top_peptide_count = top_x_peptide_counts[i]
    how_much_response_captured_in_top_peptides(complete_data, methods_all, top_peptide_count)
overall_auc_binary_classification_based(complete_data_excluding_minors, methods_mass_spec_and_netmhcpan_4_l, max_fpr_for_partial_auc)
overall_auc_binary_classification_based(complete_data_excluding_minors_and_modified, methods_all, max_fpr_for_partial_auc)

end_time = time.time()
total_time_taken = end_time - start_time
print('end\n\nTotal time taken:', str(datetime.timedelta(seconds=total_time_taken)))
