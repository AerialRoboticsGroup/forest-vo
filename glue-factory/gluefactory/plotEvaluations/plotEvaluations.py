import os
import ast 
from settings import ROOT_DIR
import matplotlib.pyplot as plt
import datetime
import numpy as np

# import scienceplots

precisionTable = False
AUCTable = True
myModels = True

if myModels:
    # Can filter the models you want that are output/training folder
    searches = [""]
else:
    # Can filter the pretraing models to include
    searches = [""]

if myModels:
    path = f"{ROOT_DIR}/evaluations/"
else:
    path = f"{ROOT_DIR}/pretrainedEvaluations/"

auc_key = 'AUC_rel_pose_error_0.5' 
# auc_key = 'AUC_mean_r_pose_error_0.5'
# auc_key = 'AUC_mean_t_pose_error_0.5'

expName = 'treeEval1_poselib.txt' 
# expName = 'finnEval_poselib.txt' 

folders = os.listdir(path)
files = []

for folder in folders:
    try:
        for file in os.listdir(path + folder): 
            if file == expName:
                files.append(f"{folder}/{file}")
    except:
        pass

fullPaths = [path + file for file in files]
print(fullPaths)

# Specify what keys you want to pull from the evaluation files
if precisionTable:
    search_lines = ['mepi_prec@1e-4', 'mepi_prec@5e-4', 'mepi_prec@1e-3', 'mepi_prec@5e-3', 'mepi_prec@1e-2', 'mransac_inl%']

if AUCTable:
    search_lines = ['AUC_rel_pose_error_0.5', 'AUC_mean_r_pose_error_0.5', 'AUC_mean_t_pose_error_0.5', 'z-time','mnum_matches', 'mnum_keypoints']


def plot_and_save_precision(table_data, expName):
    """
    Plots precision data from a table, saves the plot, and prints the save path.

    Args:
        table_data: A list of lists containing the plot data.
                    Each inner list is expected to have the format:
                    [model_name, mode, precision_string, overall_precision]
        expName: A string representing the experiment name to be included in the filename.
    """
    precision_thresholds = ['1e-4', '5e-4', '1e-3', '5e-3', '1e-2']

    for row in table_data:    
        model_name, mode, precision_string, overall_precision = row
        precision_values = [float(x) for x in precision_string.split('/')]
        # Shorten model and mode names for the legend
        model_name = model_name.title()
        mode = mode.title()

        legend_label = f"{model_name}-{mode}"
        plt.plot(precision_thresholds, precision_values, marker='o', label=legend_label)

    # Add plot labels and title
    if expName == 'treeEval1_poselib.txt':
        dataMode = "Synthetic Data"
    else:
        dataMode = "Real Data"

    if myModels:
        pretrained = "Finetuned"
    else:
        pretrained = "Pretrained"

    # plt.style.use(['science', 'notebook'])

    plt.xlabel('Precision Thresholds') 
    plt.ylabel('Precision')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, prop={'family':'monospace'}, borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Create save directory if it doesn't exist
    save_dir = f'{ROOT_DIR}/EvaluationGraphs/'
    os.makedirs(save_dir, exist_ok=True)

    # Create filename with current datetime and expName
    now = datetime.datetime.now()
    filename = f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_{expName}_precision_plot.png'
    save_path = os.path.join(save_dir, filename)

    # Save the plot
    plt.savefig(save_path)

    # Print the save path
    print(f"Plot saved to: {save_path}")



def plot_and_save_auc_bar_chart(table_data, expName, auc_key):
    """
    Plots a bar chart of AUC data from a table, saves the plot, and prints the save path.

    Args:
        table_data: A list of lists containing the plot data.
                    Each inner list is expected to have the format:
                    [model_name, mode, auc_rel_pose_error_0.5_string, 
                     auc_mean_r_pose_error_0.5_string, 
                     auc_mean_t_pose_error_0.5_string, z_time]
        expName: A string representing the experiment name to be included in the filename.
        auc_key: The key to select the AUC data to plot 
                 ('AUC_rel_pose_error_0.5', 'AUC_mean_r_pose_error_0.5', or 'AUC_mean_t_pose_error_0.5')
    """    
    # Degrees for the x-axis
    degrees = [1, 2.5, 5, 7.5, 10, 15, 20]
    # If want to use a subset of the degrees, set the corresponding mask to True
    degree_mask = [False, False, True, False, True, False, True]
    degrees = [degrees[i] for i in range(len(degrees)) if degree_mask[i]]

    # Get the index of the AUC key in the data
    auc_key_index = {
        'AUC_rel_pose_error_0.5': 0,
        'AUC_mean_t_pose_error_0.5': 1,
        'AUC_mean_r_pose_error_0.5': 2,
    }[auc_key]
    # example data from table_data
    # [['aliked', 'lightglue', '0.0006/0.0026/0.0075/0.0152/0.0242/0.0416/0.0584', '0.6240/0.8251/0.9116/0.9411/0.9558/0.9705/0.9779', '0.0006/0.0026/0.0075/0.0152/0.0242/0.0416/0.0584', 'N/A'],

    # Create the plot
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed

    # Group data by model type (LightGlue or SuperGlue)
    lightglue_data = []
    superglue_data = []
    for row in table_data:
        model_name, mode, keypoints, matches, *auc_data, _ = row
        auc_values = [float(x) for x in auc_data[auc_key_index].split('/')]
        auc_values = [auc_values[i] for i in range(len(auc_values)) if degree_mask[i]]


        # Shorten model and mode names for the legend
        short_model_name = model_name # .replace("Super", "S").replace("Glue", "G-").replace("Light", "L").replace("Point", "P-")
        short_mode = mode # .replace("rgb", "RGB").replace("stereo", "stereo").replace("gray", "gray").replace("epipolar", "epipolar").replace("homog", "homog").replace("depth", "D")
        legend_label = f"{short_model_name}-{short_mode}"  
        legend_label = f"{short_mode}"

        if "LightGlue" in mode:
            print(mode, "\n\n\n\n\n\n\n")
            lightglue_data.append((legend_label, auc_values))
        else:
            superglue_data.append((legend_label, auc_values))

     # Set up bar positions for side-by-side plotting
    all_data = lightglue_data + superglue_data
    desired_order = ["Gray", "RGB", "RGBD", "Stereo"]
    all_data = sorted(all_data, key=lambda x: desired_order.index(x[0].split()[-1]))

    # Set up bar positions for side-by-side plotting
    bar_width = 0.15 
    index = np.arange(len(degrees))

    # Total number of bars at each degree
    n_bars_per_degree = len(all_data)

    # Plot grouped bars for each model at different thresholds
    for i, (label, values) in enumerate(all_data):
        # Calculate the offset for the current bar group
        bar_positions = index + i * bar_width
        plt.bar(bar_positions, values, bar_width, label=label)

    plt.ylim(0, 1)

    # Set x-ticks to be in the middle of the groups
    plt.xticks(index + (n_bars_per_degree - 1) * bar_width / 2, degrees)

    # # Add plot labels and title
    plt.xlabel('Degrees')
    plt.ylabel(auc_key_dict[auc_key]) 
    plt.legend(loc='upper left', fontsize=11, bbox_to_anchor=(0, 1)) 
    plt.grid(True)

    # Add plot labels and title
    if expName == 'treeEval1_poselib.txt':
        dataMode = "TartanAir"
    else:
        dataMode = "FinnForest"

    if myModels:
        pretrained = "Finetuned"
    else:
        pretrained = "Pretrained"

    if datasetTitle:
        plt.title(f'{dataMode}', fontsize=18) 
        plt.tight_layout()
    else:
        plt.title(f'Precision of {pretrained} Models at Different Thresholds - {dataMode}', fontsize=18, y=1.05)  # Increased y value

    save_dir = f'{DATA_DIR}/plots'
    os.makedirs(save_dir, exist_ok=True)

    # Create filename with current datetime
    now = datetime.datetime.now()
    filename = f'pretrained_AUC-{now.strftime("%Y-%m-%d_%H-%M-%S")}.png'
    save_path = os.path.join(save_dir, filename)

    # Save the plot
    plt.savefig(save_path, dpi=300) 
    print(f"Plot saved to: {save_path}")

# plot_and_save_auc_bar_chart(filtered_table, expName, auc_key)
result_dict = {}
for file in fullPaths:
    with open(file, 'r') as f:
        for line in f:
            for search_term in search_lines:
                if search_term in line:
                    key = os.path.join(*file.split('/')[-2:]) 
                    if key not in result_dict:
                        result_dict[key] = {} 

                    # Split the line based on ':'
                    metric_name, value_str = line.strip().split(':')

                    # Convert the value to a float and round to 4 decimal places
                    try:
                        value = round(float(value_str.strip()), 4)
                    except:
                        values_list = ast.literal_eval(value_str.strip())
                        value = [round(float(val), 4) for val in values_list]
                    # Store in the dictionary
                    result_dict[key][metric_name.strip()] = value


for k, v in result_dict.items():
    for search in searches:
        if search in k:
            # print(k)
            print(k.split('/')[0], v)
            continue
    
renameDict = {
    f'sp+lg_homography/{expName}': "SuperGlue LightGlue Homography",
    ...
}

def rename_matcher(matcher):
    try:
        base_name = renameDict[matcher] 
    except:
        print("Matcher not found in dictionary", matcher)
        base_name = matcher
    return base_name  

def matcher_sort_key(row):
    matcher = row[1].lower()  

    if "lightglue" in matcher:
        base_order = 0
    elif "superglue" in matcher:
        base_order = 1
    else:
        base_order = 2  

    descriptors = ["nothing", "epipolar", "rgb", "rgbdepth", "stereo"]
    desc_order = next((i for i, desc in enumerate(descriptors) if desc in matcher), len(descriptors))  # Get descriptor order or place at the end

    return (base_order, desc_order) 

if precisionTable:
    # Process the dictionary and generate LaTeX table
    table_data = []
    for k, v in result_dict.items():
        for search in searches:
            if search in k:
                if not myModels: 
                    extractor, matcher = k.split('+') 
                    matcher = matcher.split('_')[0]
                    print(extractor, matcher)
                else:
                    # if model does not follow convention of extractor+matcher
                    extractor = "SuperPoint"
                    matcher =  rename_matcher(k)

                mepi_values = "/".join([f"{v[key]:.4f}" for key in v.keys() if 'mepi_prec' in key])
                ransac_value = f"{v['mransac_inl%']:.4f}"
                table_data.append([extractor, matcher, mepi_values, ransac_value])

    # table_data = sorted(table_data, key=matcher_sort_key)
    plot_and_save_precision(table_data, expName)

    # Create the LaTeX table string (
    makeLatex = True
    if makeLatex:
        latex_table = r"""
        \begin{table}[ht]
        \centering
        \footnotesize 
        \caption{mEpi Precision and RANSAC Inlier Percentage for Different Methods}
        \label{tab:results}
        \setlength\tabcolsep{3pt} 

        \begin{tabular}{ll>{\centering\arraybackslash}p{5cm}c} 
        \toprule
        \multicolumn{1}{c}{\textbf{Extractor}} & \multicolumn{1}{c}{\textbf{Matcher}} & \multicolumn{1}{c}{\textbf{1e-4/5e-4/1e-3/5e-3/1e-2}} & \textbf{RANSAC Inlier \%} \\ 
        \midrule
        """

        for row in table_data:
            latex_table += " & ".join(row) + r" \\" + "\n"

        latex_table += r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """

        # Print the LaTeX table
        print(latex_table)
    
if AUCTable:
    # Process the dictionary and generate LaTeX table
    table_data = []
    for k, v in result_dict.items():
        for search in searches:
            if search in k:
                # print(search, k)
                # print(v.)
                if not myModels:                
                    extractor, matcher = k.split('+') 
                    matcher = matcher.split('_')[0]
                else:
                    extractor = "SuperPoint"
                    matcher =  rename_matcher(k) 
                    print("called rename matcher")

                auc_rel_pose = "/".join([f"{float(val):.4f}" for val in v['AUC_rel_pose_error_0.5']])
                auc_trans = "/".join([f"{float(val):.4f}" for val in v['AUC_mean_t_pose_error_0.5']])
                auc_rot = "/".join([f"{float(val):.4f}" for val in v['AUC_mean_r_pose_error_0.5']])

                keypoints = v["mnum_keypoints"]
                matches = v["mnum_matches"]

                # Extract and format time 
                time_values = v.get('z-time')  
                if time_values:
                    if isinstance(time_values, list):
                        time_ms = "/".join([f"{float(val):.4f}" for val in time_values])
                    else:
                        time_ms = f"{float(time_values):.4f}"
                else:
                    time_ms = "N/A"

                table_data = sorted(table_data, key=matcher_sort_key)
                table_data.append([extractor, matcher, keypoints, matches, auc_rel_pose, auc_trans, auc_rot, time_ms]) 

    # Create the LaTeX table string
    
    plot_and_save_auc_bar_chart(table_data, expName, auc_key)
    latex_table = r"""
    \begin{table}[ht]
    \centering
    \footnotesize 
    \caption{AUC for different methods}
    \label{tab:auc_results}
    \setlength\tabcolsep{3pt} 

    % Define boolean variables for conditional inclusion
    \newboolean{showRelativeError}
    \newboolean{showTranslationalError}
    \newboolean{showRotationalError}
    \newboolean{showTiming}  % New boolean for timing

    % Set the values of the boolean variables (true or false)
    \setboolean{showRelativeError}{true}     % Show Relative Error
    \setboolean{showTranslationalError}{false}  % Hide Translational Error  
    \setboolean{showRotationalError}{false}    % Hide Rotational Error
    \setboolean{showTiming}{true}              % Show Timing

    %  \begin{tabular}{l>{\centering\arraybackslash}m{2cm}cccc} 
    % \begin{tabular}{l>{\centering\arraybackslash}m{2cm}cccccc} 
    % \toprule
    % \multicolumn{2}{c}{\textbf{Extractor Matcher}} & 
    % \multicolumn{4}{c}{\textbf{Extractor Matcher KeyPoints Matches}} & 
    % \ifthenelse{\boolean{showRelativeError}}{\textbf{Relative Error}}{} & 
    % \ifthenelse{\boolean{showTranslationalError}}{\textbf{Translational Error}}{} & 
    % \ifthenelse{\boolean{showRotationalError}}{\textbf{Rotational Error}}{} & 
    % \ifthenelse{\boolean{showTiming}}{\textbf{Time /ms}}{} \\
    
    \begin{tabular}{l>{\centering\arraybackslash}m{2cm}cccccc} 
    \toprule
    % \multicolumn{4}{c}{\textbf{Extractor Matcher KeyPoints Matches}} & 
    \multirow{2}{*}{\textbf{Extractor}} & \multirow{2}{*}{\textbf{Matcher}} & \multirow{2}{*}{\textbf{KeyPoints}} & \multirow{2}{*}{\textbf{Matches}} & 
    \ifthenelse{\boolean{showRelativeError}}{\multicolumn{1}{c}{\textbf{LO-RANSAC AUC}}}{} & 
    \ifthenelse{\boolean{showTranslationalError}}{\textbf{Translational Error}}{} & 
    \ifthenelse{\boolean{showRotationalError}}{\textbf{Rotational Error}}{} & 
    \ifthenelse{\boolean{showTiming}}{\textbf{Time /ms}}{} \\
    \cmidrule(lr){5-5}
    & & & & \multicolumn{1}{c}{\textbf{5°/10°/20°}} \\
    \midrule
    """
    #  \multicolumn{2}{c}{\textbf{Extractor Matcher}} & \ifthenelse{\boolean{showRelativeError}}{\multicolumn{1}{>{\centering\arraybackslash}m{3.5cm}}{\textbf{Relative Error}}}{}& \ifthenelse{\boolean{showTranslationalError}}{\multicolumn{1}{>{\centering\arraybackslash}m{3.5cm}}{\textbf{Translational Error}}}{}& \ifthenelse{\boolean{showRotationalError}}{\multicolumn{1}{>{\centering\arraybackslash}m{3.5cm}}{\textbf{Rotational Error}}}{}& \ifthenelse{\boolean{showTiming}}{\multicolumn{1}{c}{\textbf{Time /ms}}}{} \\ 
   
    for row in table_data:
        extractor, matcher, keypoints, matches, auc_rel_pose, auc_trans, auc_rot, time_ms = row
        latex_table += f"{extractor} & {matcher} & {keypoints} & {matches} "
        latex_table += f"& \\ifthenelse{{\\boolean{{showRelativeError}}}}{{{auc_rel_pose}}}{{}} "
        latex_table += f"& \\ifthenelse{{\\boolean{{showTranslationalError}}}}{{{auc_trans}}}{{}} "
        latex_table += f"& \\ifthenelse{{\\boolean{{showRotationalError}}}}{{{auc_rot}}}{{}} "
        latex_table += f"& \\ifthenelse{{\\boolean{{showTiming}}}}{{{time_ms}}}{{}} \\\\" + "\n"

    latex_table += r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """


    print(latex_table)


print("Evaluation complete for ", expName)

