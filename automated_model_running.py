import os
import sys
from all_model_configs import *
from itertools import product
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# automated model configuration running:

def criterion_from_criterion_name(criterion_name, target_metric): #since the criterion in the configuration excel is string, convert to func
    MCID_dict = {
    'phq9': 5,
    'gad7': 4
    }

    if criterion_name == "none":
        return lambda x : x
    elif criterion_name == "category":
        if "phq9" in target_metric:
            return categorize_phq9
        elif "gad7" in target_metric:
            return categorize_gad7
        elif "qids" in target_metric:
            return categorize_qids
    else:
        threshold = criterion_name.split(">=")[1]
        if threshold == "mcid":
            return lambda x : abs(x) >= MCID_dict[target_metric]
        elif threshold.isnumeric():
            return lambda x : x >= float(threshold)

def combs_list(participant_id, framework_excel): #i know all configs will be tried with both rf and xgboost so no need for another col
    path = "/data/benlab/users/dylan_saiyara/"
    df = pd.read_excel(f"{path}/{framework_excel}", sheet_name=participant_id)
    ps = df["ps"].dropna()
    target_q = df["target_q"].dropna()
    target_type = df["target_type"].dropna()
    cat_or_reg = df["cat_or_reg"].dropna()
    window_size = df["window_size"].dropna()
    is_sliding = df["is_sliding"].dropna()
    criterion = df["criterion"].dropna()

    return list(product( target_q, target_type, ps, cat_or_reg, window_size, is_sliding, criterion))

def fill_model_configs(participant_id, framework_excel): #fill up all model configuration excel sheet before
    #automatically running models
    
    model_configs = pd.DataFrame(columns=["Participant", "Target metric", "Target types",	"Predictor set", "Window size", "Sliding?",	"Categorization or regression",	"Categorization criterion",	"Datapoints",	"Accuracy",	"F1",	"AUC",	"shap_plot_loc"	])
    
    all_combinations = combs_list(participant_id, framework_excel)
    
    for i, combination in enumerate(all_combinations):
        target_q, target_type, ps, cat_or_reg, window_size, is_sliding, criterion = combination
        
        row = [participant_id, target_q, target_type, ps, window_size, is_sliding, cat_or_reg, criterion, -1, -1, -1, -1, ""]
        
        model_configs.loc[len(model_configs)] = row

    
    return model_configs
    #.to_excel(f"{path}{participant_id}_model_configurations.xlsx")

    #add to previous results instead of erasing and rerunning


def filtering_ps_helper(participant_id, framework_excel): #filter out ps and target rows, from participant_framework or participant_framework_addtional
    model_configs = fill_model_configs(participant_id, framework_excel)

    phq9_rows = model_configs["Target metric"] == "phq9"
    gad7_rows = model_configs["Target metric"] == "gad7"
    qids_rows = model_configs["Target metric"] == "qids"
    cssrs_rows = model_configs["Target metric"] == "cssrs"
    lshs_rows = model_configs["Target metric"] == "lshs"
    ahrs_rows = model_configs["Target metric"] == "ahrs"
    pdi_rows = model_configs["Target metric"] == "pdi"

    redcap_q_rows = phq9_rows | gad7_rows | qids_rows | cssrs_rows | lshs_rows | ahrs_rows | pdi_rows

    ema_mood = model_configs["Target metric"]=="mood_ema"
    ema_energy = model_configs["Target metric"]=="\"Rate your energy level\""
    ema_rows = ema_mood | ema_energy

    if framework_excel == "participant_framework_ind.xlsx":
        #cond 1: redcap questionnaire rows (phq9/gad7/etc) that have only predictor set 1, 1.5, 1.75, 2.5, 3.5, 4.5, 5.5, 6
        
        
        ps1_rows = model_configs["Predictor set"] == 1
        ps1_prime_rows = model_configs["Predictor set"] == 1.5
        ps1_prime_2_rows = model_configs["Predictor set"] == 1.75
        
        ps2_prime_rows = model_configs["Predictor set"] == 2.5

        ps3_prime_rows = model_configs["Predictor set"] == 3.5
        ps4_prime_rows = model_configs["Predictor set"] == 4.5
        ps5_prime_rows = model_configs["Predictor set"] == 5.5

        ps6_rows =  model_configs["Predictor set"] == 6

        redcap_ps = ps1_rows | ps1_prime_rows | ps1_prime_2_rows | ps2_prime_rows | ps3_prime_rows | ps4_prime_rows | ps5_prime_rows | ps6_rows
        
        cond_1 = (redcap_q_rows & redcap_ps)
            
        #cond 2: ema rows that have only predictor set 2, 3, 4, 5 -> for 016, 022; rate your energy level rows with predictor set 2 -> 007
            
        ps2_rows = model_configs["Predictor set"] == 2
        ps3_rows = model_configs["Predictor set"] == 3
        ps4_rows = model_configs["Predictor set"] == 4
        ps5_rows = model_configs["Predictor set"] == 5

        ema_ps = ps2_rows | ps3_rows | ps4_rows | ps5_rows
        
        cond_2 = ema_rows & ema_ps
        

    elif framework_excel == "participant_framework_agg.xlsx":
        #cond 1: redcap questionnaire rows (phq9/gad7/etc) that have only predictor set 7.5, 8.5, 9.5, 10.5

        ps7_prime_rows = model_configs["Predictor set"] == 7.5
        ps_8_prime_rows = model_configs["Predictor set"] == 8.5
        ps_9_prime_rows = model_configs["Predictor set"] == 9.5
        ps_10_prime_rows = model_configs["Predictor set"] == 10.5

        redcap_ps = ps7_prime_rows | ps_8_prime_rows | ps_9_prime_rows | ps_10_prime_rows

        cond_1 = (redcap_q_rows & redcap_ps)
                  
        #cond 2: ema rows that have only predictor set 7, 8, 9, 10

        ps7_rows = model_configs["Predictor set"] == 7
        ps_8_rows = model_configs["Predictor set"] == 8
        ps_9_rows = model_configs["Predictor set"] == 9
        ps_10_rows = model_configs["Predictor set"] == 10
        
        ema_ps = ps7_rows | ps_8_rows | ps_9_rows | ps_10_rows

        cond_2 = (ema_rows & ema_ps)

    return cond_1, cond_2, ema_rows

def filter_out_mismatched_rows(participant_id, framework_excel):
    model_configs = fill_model_configs(participant_id, framework_excel)
    #filter out mismatched rows, like ps 2 and phq9/gad7

    cond_1, cond_2, ema_rows = filtering_ps_helper(participant_id, framework_excel)
    
    #cond 3: for classification rows, make sure they don't have "none" as criterion
    classification_rows = model_configs["Categorization or regression"]=="classification"
    not_none_criteria = model_configs["Categorization criterion"]!="none"
    cond_3 = classification_rows & not_none_criteria

    #cond_4: or regression rows, make sure they have "none" as criterion
    regression_rows = model_configs["Categorization or regression"]=="regression"
    none_criteria = model_configs["Categorization criterion"]=="none"
    cond_4 = regression_rows & none_criteria

    #cond_5: diff prev and diff baseline shouldnt have >= 10 criterion and category of phq9/gad7/qids as criterion
    diff_rows = (model_configs["Target types"]=="diff_prev") | (model_configs["Target types"]=="diff_baseline")
    criteria_for_diff_rows = (model_configs["Categorization criterion"]!=">=10") & (model_configs["Categorization criterion"]!="category")
    cond_5 = diff_rows & criteria_for_diff_rows

    #cond_6: score shouldnt have >=0 and mcid-based criterion 
    score_rows = model_configs["Target types"]=="score"
    criteria_for_score_rows = (model_configs["Categorization criterion"]!=">=mcid") & (model_configs["Categorization criterion"]!=">=0")
    cond_6 = score_rows & criteria_for_score_rows

    #i apply them later
    #cond_7: lshs, ahrs, cssrs, pdi, and ema score rows shouldnt have classification
    cssrs_rows = model_configs["Target metric"] == "cssrs"
    lshs_rows = model_configs["Target metric"] == "lshs"
    ahrs_rows = model_configs["Target metric"] == "ahrs"
    pdi_rows = model_configs["Target metric"] == "pdi"
    ema_redcap_score_rows = (ema_rows | cssrs_rows | lshs_rows | ahrs_rows | pdi_rows) & score_rows

    ema_score_with_classification_rows  = ema_redcap_score_rows & classification_rows

    #cond_8: lshs, ahrs, cssrs, pdi and ema classification rows shouldn't have mcid criterion
    ema_redcap_classification_rows = (ema_rows | cssrs_rows | lshs_rows | ahrs_rows | pdi_rows) & classification_rows
    mcid_rows = model_configs["Categorization criterion"]==">=mcid"
    ema_classification_rows_with_mcid = ema_redcap_classification_rows & mcid_rows

    #cond 9: qids shouldn't have >=10 categorization or mcid
    qids_rows = model_configs["Target metric"] == "qids"
    greater_than_10_rows = model_configs["Categorization criterion"] == ">=10"
    mcid_rows =  model_configs["Categorization criterion"] == ">=mcid"

    qids_without_greater_than_10_or_mcid_rows = qids_rows & (greater_than_10_rows | mcid_rows)

    #does this only work because the conditions are mutually exclusive?
    selected_rows = model_configs[((cond_1) | (cond_2)) & (cond_3 | cond_4) & (cond_5 | cond_6)]
    
    selected_rows = selected_rows.drop(selected_rows[ema_score_with_classification_rows].index)
    selected_rows = selected_rows.drop(selected_rows[ema_classification_rows_with_mcid].index)
    selected_rows = selected_rows.drop(selected_rows[qids_without_greater_than_10_or_mcid_rows].index)
    '''
    model_configs[((((model_configs["Target metric"]=="phq9") | (model_configs["Target metric"]=="gad7") | (model_configs["Target metric"]=="summed_phq9_gad7"))
                & ((model_configs["Predictor set"] != 2))) | (((model_configs["Target metric"]=="mood_ema") | (model_configs["Target metric"]=="\"Rate your energy level\"")) & (model_configs["Predictor set"] == 2)))
                
                & (((model_configs["Categorization or regression"]=="classification") & (model_configs["Categorization criterion"]!="none"))
                | ((model_configs["Categorization or regression"]=="regression") & (model_configs["Categorization criterion"]=="none")))

                & ((model_configs["Target types"]=="score") & ((model_configs["Categorization criterion"]!=">=mcid") & (model_configs["Categorization criterion"]!=">=0")) |
                   ((model_configs["Target types"]=="diff_prev")| (model_configs["Target types"]=="diff_baseline")) & ((model_configs["Categorization criterion"]!=">=10") & (model_configs["Categorization criterion"]!="category"))
                   )
                    ]
    '''
    
    return selected_rows

#compute number of datapoints and label distribution on a subset of model_configs and save it
def num_datapoints_and_label_dist(participant_id, model_configs):
    #model_configs = filter_out_mismatched_rows(participant_id, framework_excel)
    
    for_datapoints = model_configs[['Participant', 'Target metric', 'Target types', 'Predictor set', 'Window size', 'Sliding?']].drop_duplicates()

    #same X and raw y for a lot of configs that then apply categorization criterion on y

    num_datapoints_df = pd.DataFrame(columns=["Target metric", "Target types", "Predictor set", "Window size", "Sliding?", "Categorization or regression", "Categorization criterion", "Datapoints", "Label distribution"])
    '''
    gps_df = load_gps_data(participant_id)
    if len(gps_df) != 0:
        t_with_stay, staypoints = get_pf_and_staypoints(gdf=gps_df, method_for_home_compute="OSNA")
        generated_trips = get_trips(t_with_stay, staypoints)
    else:
        staypoints=None
        generated_trips=None
    '''

    for i in range(len(for_datapoints)):
            target_metric = for_datapoints["Target metric"].iloc[i]
            target_type = for_datapoints["Target types"].iloc[i] #score, diff prev/baseline
            target = target_metric
            if target_type != "score":
                target += f"_{target_type}" 
    
            #config = model_configs["Configuration"].iloc[i]
            ps = for_datapoints["Predictor set"].iloc[i]
            window_size = for_datapoints["Window size"].iloc[i]
            is_sliding = bool(for_datapoints["Sliding?"].iloc[i])
            
            X,y, features = get_dataset(participant_id, target, window_size, is_sliding, ps)
            
            merged_back_with_model_configs = pd.merge(
                left=for_datapoints.iloc[i: i+1], 
                right=model_configs,
                how='left',
                left_on=['Participant', 'Target metric', 'Target types', 
                'Predictor set', 'Window size', 'Sliding?'],
                right_on=['Participant', 'Target metric', 'Target types',
                'Predictor set', 'Window size', 'Sliding?'],
            )
            
            for j in range (len(merged_back_with_model_configs)):
                reg_or_cat = merged_back_with_model_configs["Categorization or regression"].iloc[j] 
                criterion_name = merged_back_with_model_configs["Categorization criterion"].iloc[j] #create function from string
                criterion = criterion_from_criterion_name(criterion_name, target_metric)
                
                if reg_or_cat == "classification":     
                    y_cat = list(map(criterion, y))
                    y_cat = pd.DataFrame(y_cat)
                    label_dist = y_cat.value_counts().reset_index()
                    label_dist["percentage"] = label_dist["count"].apply(lambda x : x/label_dist["count"].sum()).apply(round, args=(2,)).apply(lambda x : f"{x}")
                    l = label_dist["percentage"].to_list()
                    l = ', '.join(l)
                   
                else:
                    l = "None"
                
                print(target, ps, window_size, is_sliding, reg_or_cat, criterion_name, X.shape[0], l)
                
                row = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, X.shape[0], l]
                num_datapoints_df.loc[len(num_datapoints_df)] = row
    '''
    if framework_excel == "participant_framework_ind.xlsx":
        num_datapoints_df.to_csv(f"/data/benlab/users/dylan_saiyara/{participant_id}_config_num_datapoints_ind.csv", index=False)

    elif framework_excel == "participant_framework_agg.xlsx":
        num_datapoints_df.to_csv(f"/data/benlab/users/dylan_saiyara/{participant_id}_config_num_datapoints_agg.csv", index=False)
    '''
    return num_datapoints_df

#failsafe given running so many configs at once is risky
                                                    #results is a csv string
def write_config_results_to_file(participant_id, config_name, model, results):
    results = ','.join(results)
    path="/data/benlab/users/alice_saiyara/shap_plots"
    with open(f"{path}/{participant_id}_{config_name}_{model}/{participant_id}_{config_name}_{model}", 'w') as fp:
                    fp.write(results)
    fp.close()

                    #selected model configs based on number of datapoints and distribution of labels for classification
                                #pass in model configs df with num datapoints and dist of labels, which is model agnostic
def run_configs(participant_id, model_configs):
    '''
    gps_df = load_gps_data(participant_id)
    
    if len(gps_df) != 0:
        t_with_stay, staypoints = get_pf_and_staypoints(gdf=gps_df, method_for_home_compute="OSNA")
        generated_trips = get_trips(t_with_stay, staypoints)
    else:
        staypoints=None
        generated_trips=None
    '''

    results_df = pd.DataFrame(columns=["Target metric", "Target types", "Predictor set", "Window size", "Sliding?", "Categorization or regression", "Categorization criterion", "Pred type", "Datapoints", "Label distribution"
                                       , "Model", "mean acc", "sd acc", "mean auc", "sd auc", "mean f1", "sd f1", "mean sensitivity", "sd sensitivity", "mean specificity", "sd specificity", "mean npv", "sd npv", "mean ppv", "sd ppv", "mean mse", "sd mse", "mean r_squared", "sd r_squared", "train_error",
                                       "shap_plot_loc"
                                       ]) #also add pred type in results!
    for i in range(len(model_configs)):
        target_metric = model_configs["Target metric"].iloc[i]
        target_type = model_configs["Target types"].iloc[i] #score, diff prev/baseline
        target = target_metric
        if target_type != "score":
            target += f"_{target_type}"

        ps = model_configs["Predictor set"].iloc[i]
        window_size = model_configs["Window size"].iloc[i]
        is_sliding = bool(model_configs["Sliding?"].iloc[i])
        reg_or_cat = model_configs["Categorization or regression"].iloc[i] #i haven't implemented regression
        criterion_name =  model_configs["Categorization criterion"].iloc[i] #create function from string
        
        criterion = criterion_from_criterion_name(criterion_name, target_metric)
        
        config_name = f"target:{target}, ps:{ps}, window_size:{window_size}, is_sliding:{is_sliding}, reg_or_cat:{reg_or_cat}, criterion:{criterion_name}"

        X,y, features = get_dataset(participant_id, target, window_size, is_sliding, ps)
        
        #prior to running classification models make sure there is proper dist of labels
        #like at least 20-80 for binary? what about multiclass?
        num_datapoints = model_configs["Datapoints"].iloc[i]
        print(target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, num_datapoints)
        if num_datapoints > 5:
            if reg_or_cat == "classification": 
                label_dist = list(map(float, model_configs["Label distribution"].iloc[i].split(", "))) #string, converted to list
                over_80_percent = list(map(lambda x : x > 0.8, label_dist))
            
                if (len(label_dist) == 1 ) or (True in over_80_percent): #only one category present in labels according to criterion
                                                        #or one category is there 80% time according to criterion
        
                    row_rf = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, "none", num_datapoints, model_configs["Label distribution"].iloc[i], "rf", -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, "none"]
                    row_xgboost = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, "none", num_datapoints, model_configs["Label distribution"].iloc[i], "xgboost", -1, -1, -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, "none" ]
                    results_df.loc[len(results_df)] = row_rf
                    results_df.loc[len(results_df)] = row_xgboost
                    print("SKIPPED CLASSIFICATION")
                    continue

                else:
                    if len(label_dist) == 2: #binary prediction
                        pred_type = "binary"
                        
                    else: #multi class prediction
                        pred_type = "multi"
                    
                    rf_mean_acc, rf_sd_acc, rf_mean_auc, rf_sd_auc, rf_mean_f1, rf_sd_f1, rf_mean_sensitivities, rf_sd_sensitivities, rf_mean_specificities, rf_sd_specificities, rf_mean_npvs, rf_sd_npvs, rf_mean_ppvs, rf_sd_ppvs, rf_shaps, rf_train_errors, pred_type, shap_loc_rf = avg_cross_val_auc_rf(X=X,y=np.array(y), features=features,  criterion=criterion, shap_plot_loc=f"{participant_id}_{config_name}_rf", shap_plot_name=f"{participant_id}_{config_name}_rf")
                    row_rf = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, pred_type, num_datapoints, model_configs["Label distribution"].iloc[i], "rf", rf_mean_acc, rf_sd_acc, rf_mean_auc, rf_sd_auc, rf_mean_f1, rf_sd_f1, rf_mean_sensitivities, rf_sd_sensitivities, rf_mean_specificities, rf_sd_specificities, rf_mean_npvs, rf_sd_npvs, rf_mean_ppvs, rf_sd_ppvs, -1, -1, -1, -1, rf_train_errors, shap_loc_rf ]
                    #write_config_results_to_file(participant_id, config_name, "rf", tuple(list(map(str, row_rf))))
                    results_df.loc[len(results_df)] = row_rf
                    xg_mean_acc, xg_sd_acc, xg_mean_auc, xg_sd_auc, xg_mean_f1, xg_sd_f1, xg_mean_sensitivities, xg_sd_sensitivities, xg_mean_specificities, xg_sd_specificities, xg_mean_npvs, xg_sd_npvs, xg_mean_ppvs, xg_sd_ppvs, xg_shaps,  xg_train_errors, pred_type, shap_loc_xg= avg_cross_val_auc_xgboost(X=X, y=np.array(y), features=features,  criterion=criterion, shap_plot_loc=f"{participant_id}_{config_name}_xgboost", shap_plot_name=f"{participant_id}_{config_name}_xgboost")
                                                                            # binary or multiclass pred
                        #Xgboost needs to know if it's binary or multiclass prediction, check dist of labels
                    
                    row_xgboost = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, pred_type, num_datapoints, model_configs["Label distribution"].iloc[i], "xgboost", xg_mean_acc, xg_sd_acc, xg_mean_auc, xg_sd_auc, xg_mean_f1, xg_sd_f1, xg_mean_sensitivities, xg_sd_sensitivities, xg_mean_specificities, xg_sd_specificities, xg_mean_npvs, xg_sd_npvs, xg_mean_ppvs, xg_sd_ppvs, -1, -1, -1, -1,  xg_train_errors, shap_loc_xg]
                    results_df.loc[len(results_df)] = row_xgboost               #result
                    #write_config_results_to_file(participant_id, config_name, "xgboost", tuple(list(map(str, row_xgboost))))
                    
            else: #regression
                rf_mse, rf_sd_mse, rf_mean_r_squared, rf_sd_r_squared, rf_shaps, rf_train_errors, shap_loc_rf = avg_cross_val_regression(5, X, y, features, model="rf", model_max_depth=5, shap_plot_loc=f"{participant_id}_{config_name}_rf", shap_plot_name=f"{participant_id}_{config_name}_rf")
                row_rf = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, "none", num_datapoints, model_configs["Label distribution"].iloc[i], "rf", -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, rf_mse, rf_sd_mse, rf_mean_r_squared, rf_sd_r_squared, rf_train_errors, shap_loc_rf]
                results_df.loc[len(results_df)] = row_rf
                #write_config_results_to_file(participant_id, config_name, "rf", tuple(list(map(str, row_rf))))

                xgboost_mse, xg_sd_mse, xg_mean_r_squared, xg_sd_r_squared, xgboost_shaps, xg_train_errors, shap_loc_xg = avg_cross_val_regression(5, X, y, features, model="xgboost", model_max_depth=5, shap_plot_loc=f"{participant_id}_{config_name}_xgboost", shap_plot_name=f"{participant_id}_{config_name}_xgboost")
                row_xgboost = [target_metric, target_type, ps, window_size, is_sliding, reg_or_cat, criterion_name, "none", num_datapoints, model_configs["Label distribution"].iloc[i], "xgboost", -1, -1, -1, -1, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1, xgboost_mse, xg_sd_mse, xg_mean_r_squared, xg_sd_r_squared, xg_train_errors, shap_loc_xg]
                                                                                                                                                                                # ^
                                                                                                                                                                                # the txt file inside shap plots for xgboost-based regression mistakenly have rf as model, inside the file             
                results_df.loc[len(results_df)] = row_xgboost
                #write_config_results_to_file(participant_id, config_name, "xgboost", tuple(list(map(str, row_xgboost))))

    return results_df