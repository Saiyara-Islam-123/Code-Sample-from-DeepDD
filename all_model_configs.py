import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,  roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from sklearn.metrics import mean_squared_error
from generate_window import *
from python.loading_duckdb import *
from predictors import *
from target_metrics import *
from target_types import *
import xgboost as xgb
from params_from_yaml import *
import shap
import matplotlib.pyplot as pl
from sklearn.metrics import r2_score
from features_list import *

################################## for both classification and regression #####################

##################################### GeneActiv features ###################################
#for ps1 variants
def ps1_helper(redcap_id, sleep_act_features_csv, target):
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    df_features = get_sleep_act_features_df(processed_file=sleep_act_features_csv)
    df_target = redcap_target(target, redcap_id)    
    
    df_features = date_features(df_features, participant_id)
    return df_features, df_target


#returns the averaged raw targets across a window then apply category if needed later
def get_dataset_redcap_ps1(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv):
    df_features, df_target = ps1_helper(redcap_id, sleep_act_features_csv, target)
    #date encoding is down in the window function
    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
                
    return X,y, df_features.drop(columns=["date"]).columns #no staypoints or trips
                #i need feature names for shap plots


#activity only to predict redcap q
def get_dataset_redcap_ps1_prime(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv):
    df_features, df_target = ps1_helper(redcap_id, sleep_act_features_csv, target)
    
    df_features.drop(columns=['Sleep', "Sleep.Onset.Time", "Rise.Time", "Total.Elapsed.Bed.Time",
                            "Total.Sleep.Time", "Sleep.Efficiency", "Num.Active.Periods",
                            "Median.Activity.Length"
                              

                              ], inplace=True) #drop list of sleep features

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
                    
    return X,y, df_features.drop(columns=["date"]).columns


#sleep only to predict redcap q
def get_dataset_redcap_ps1_prime_2(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv):
    df_features, df_target = ps1_helper(redcap_id, sleep_act_features_csv, target)
    df_features.drop(columns=["Steps", 'Sedentary', "Light", "Moderate", "Vigorous", "Total.Wake.Time"

                              ], inplace=True) #drop list of sleep features

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
    
    return X,y, df_features.drop(columns=["date"]).columns

#light and temp to predict redcap q
def get_dataset_redcap_ps6(target, target_col, window_size, is_sliding, redcap_id, light_temp_csv):
    df_features = get_light_temp_features(light_temp_csv)
    df_target = redcap_target(target, redcap_id)    
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    df_features = date_features(df_features, participant_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
    return X,y, df_features.drop(columns=["date"]).columns

####################################### APP FEATURES ###################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Need to Update ~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_dataset_ema_ps2(target, window_size, is_sliding, ema_csv, redcap_id):
    #aggregate call and gps features within a window. Get raw data first
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    call_df = load_call_data(participant_id)
    call_df = call_df.rename(columns={"timestamp": "timestamp_calls"})
    
    ema_df = load_ema_df(target, ema_csv)

    df_features = date_features(call_df, participant_id)
                
    X,y = window(df_features, ema_df, None, window_size, is_sliding)
    return X,y, gps_features_list + ["total incoming", "incoming mean duration", "incoming duration sd", "total outgoing", "outgoing mean duration", "outgoing duration sd", "all_calls_total_duration", "all_calls_mean_duration", "all_calls_duration_sd", "total missed",
                     "date", "is_holiday", "completed_assess"
    
                     ]


def get_dataset_redcap_ps2_prime(target, target_col, window_size, is_sliding, redcap_id): #predicting redcap q based outcomes using app data, and the others
    
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    
    call_df = load_call_data(participant_id)
    call_df = call_df.rename(columns={"timestamp": "timestamp_calls"})
    
    #Computed staypoints and trips right away    
    
    df_features = date_features(call_df, participant_id)
    #ema_df = interpolate_ema()[["date", "Rate your energy level."]] #for the time being I'll use "Rate your energy level" alone
    #df_features = pd.merge(df_features, ema_df, on="date", how="inner") #this is PS2_prime

    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
    return X,y, gps_features_list + ["total incoming", "incoming mean duration", "incoming duration sd", "total outgoing", "outgoing mean duration", "outgoing duration sd", "all_calls_total_duration", "all_calls_mean_duration", "all_calls_duration_sd", "total missed",
                     "date", "is_holiday", "completed_assess"
    
                     ]

# app actigraphy to predict emas
def get_dataset_ema_ps3(target, window_size, is_sliding, participant_id, app_act_csv, ema_csv):
    app_act = app_activity_features(app_act_csv)
    
    df_features = date_features(app_act, participant_id)

    ema_df = load_ema_df(target, ema_csv)

    X,y = window(df_features, ema_df, None, window_size, is_sliding)
    
    return X,y, df_features.drop(columns=["date"]).columns

# app actigraphy to predict redcap qs
def get_dataset_redcap_ps3_prime(target, target_col, window_size, is_sliding, redcap_id, app_act_csv):
    app_act = app_activity_features(app_act_csv)    
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    df_features = date_features(app_act, participant_id)

    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
        
    return X,y, df_features.drop(columns=["date"]).columns

#~~~~~~~~~~~~~~~~~~~ Need to update~~~~~~~~~~~~~~~~~~~~~~~~~~~
#gps alone as predictor
def get_dataset_ema_ps4(target, window_size, is_sliding, ema_csv, participant_id):
    ema_df = load_ema_df(target, ema_csv)
    df_features = date_features(pd.DataFrame(columns=["date"]), participant_id)

    X,y = window(df_features, ema_df, None, window_size, is_sliding)
            
    return X,y, gps_features_list + ["date", "is_holiday", "completed_assess"]

def get_dataset_redcap_ps4_prime(target, target_col, window_size, is_sliding, redcap_id):
    participant_id = get_participant_id_from_redcap_id(redcap_id)

    df_features = date_features(pd.DataFrame(columns=["date"]), participant_id)
    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)

    return X,y, gps_features_list + ["date", "is_holiday", "completed_assess"]



#screen time to predict ema outcomes
def get_dataset_ema_ps5(target, window_size, is_sliding, ema_csv, participant_id, screen_time_csv):
    ema_df = load_ema_df(target, ema_csv)
    df_features = get_screen_time_features(screen_time_csv)
    df_features = date_features(df_features, participant_id)

    X,y = window(df_features, ema_df, None, window_size, is_sliding,)
    return X,y, df_features.drop(columns=["date"]).columns

#screen time to predict redcap outcomes
def get_dataset_redcap_ps5_prime(target, target_col, window_size, is_sliding, redcap_id, screen_time_csv):
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    df_features = get_screen_time_features(screen_time_csv)
    df_features = date_features(df_features, participant_id)
    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding, )
    
    return X,y, df_features.drop(columns=["date"]).columns

########################################## Aggregate predictor sets ########################################################

def get_all_geneactiv_features(sleep_act_csv, light_temp_csv, participant_id):
    sleep_act = get_sleep_act_features_df(processed_file=sleep_act_csv)
    light_temp = get_light_temp_features(light_temp_csv)
    df_features = pd.merge(sleep_act, light_temp, on="date", how="inner")
    df_features = date_features(df_features, participant_id)
    return df_features

#all geneactiv features to predict emas
def get_dataset_ema_ps7(target, window_size, is_sliding, participant_id, sleep_act_csv, light_temp_csv, ema_csv):
    df_features = get_all_geneactiv_features(sleep_act_csv, light_temp_csv, participant_id)

    ema_df = load_ema_df(target, ema_csv)

    X,y = window(df_features, ema_df, target_col=None, window_len=window_size, is_sliding=is_sliding)
    return X,y, df_features.drop(columns=["date"]).columns

#all geneactiv features to predict redcap qs
def get_dataset_redcap_ps7_prime(target, target_col, window_size, is_sliding, redcap_id, sleep_act_csv, light_temp_csv):
    participant_id = get_participant_id_from_redcap_id(redcap_id)
    df_features = get_all_geneactiv_features(sleep_act_csv, light_temp_csv, participant_id)

    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding,)
    return X,y, df_features.drop(columns=["date"]).columns

#all app except GPS to predict ema
def get_features_ps8(participant_id, app_act_csv, call_csv, screen_time_csv):
    call_df = get_call_features(call_csv)
    
    app_act = app_activity_features(app_act_csv)
    screen_time_df = get_screen_time_features(screen_time_csv)
            
    df_features = pd.merge(call_df, app_act, on="date", how='inner')
    df_features = pd.merge(df_features,screen_time_df, on="date", how='inner')
    df_features = date_features(df_features, participant_id)
    
    return df_features, df_features.drop(columns=["date"]).columns

def get_dataset_ema_ps8(target, window_size, is_sliding, participant_id, app_act_csv, call_csv, screen_time_csv, ema_csv):
    df_features, features_list = get_features_ps8(participant_id, app_act_csv,  call_csv, screen_time_csv)

    ema_df = load_ema_df(target, ema_csv)

    X,y = window(df_features, ema_df, None, window_size, is_sliding)
    return X,y, features_list

def get_dataset_redcap_ps8_prime(target, target_col, window_size, is_sliding, participant_id, redcap_id, app_act_csv, call_csv, screen_time_csv):
    df_features, features_list = get_features_ps8(participant_id, app_act_csv, call_csv, screen_time_csv)
    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
    return X,y, features_list

#~~~~~~~~~~~~~~~~~~~~~~ need to update
#all app features
def get_dataset_ema_ps9(target, window_size, is_sliding, participant_id, app_act_csv, ema_csv):
    df_features, features_list = get_features_ps8(participant_id, app_act_csv)
    ema_df = load_ema_df(target, ema_csv)
    
    X,y = window(df_features, ema_df, None, window_size, is_sliding)
    return X,y, gps_features_list + features_list

def get_dataset_redcap_ps9_prime(target, target_col, window_size, is_sliding, participant_id, redcap_id, app_act_csv):
    df_features, features_list = get_features_ps8(participant_id, app_act_csv)
    df_target = redcap_target(target, redcap_id)
    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
    return X,y, gps_features_list + features_list

############################################# both geneactiv and app ###############################################
#~~~~~~~~~~~~~~~~~~~~~~ need to update
def get_dataset_ema_ps10(target, window_size, is_sliding, participant_id, sleep_act_csv, light_temp_csv, app_act_csv, ema_csv):
    geneactiv_features = get_all_geneactiv_features(sleep_act_csv, light_temp_csv, participant_id).drop(columns=["is_holiday", "completed_assess"])
    app_features, app_feautures_list_without_gps = get_features_ps8(participant_id, app_act_csv) #already has date features
    df_features = pd.merge(geneactiv_features, app_features, on="date", how="inner")
    
    ema_df = load_ema_df(target, ema_csv)

    X,y = window(df_features, ema_df, None, window_size, is_sliding)
    return X,y, list_of_geneactiv_features + gps_features_list + app_feautures_list_without_gps

def get_dataset_ema_ps10_prime(target, target_col, window_size, is_sliding, participant_id, redcap_id, sleep_act_csv, light_temp_csv, app_act_csv):
    geneactiv_features = get_all_geneactiv_features(sleep_act_csv, light_temp_csv, participant_id).drop(columns=["is_holiday", "completed_assess"])
    app_features, app_feautures_list_without_gps = get_features_ps8(participant_id, app_act_csv) #already has date features
    df_features = pd.merge(geneactiv_features, app_features, on="date", how="inner")

    df_target = redcap_target(target, redcap_id)

    X,y = window(df_features, df_target, target_col, window_size, is_sliding)
    return X,y, list_of_geneactiv_features + gps_features_list + app_feautures_list_without_gps

######################################### generic get_dataset func for all configs ###########################################

def get_dataset(participant_id, target, window_size, is_sliding, ps):
    d = get_param(participant_id)
    redcap_id = d["redcap_id"]
    sleep_act_features_csv= d["sleep_act_csv"]
    ema_csv=d["ema_csv"] 
    app_act_csv = d["app_act_csv"]
    light_temp_csv = d["light_temp_csv"]
    call_csv = d["call_csv"]
    screen_time_csv = d["screen_time_csv"]

    target_splitted = target.split("_")[0]
    
    if target_splitted in ["phq9", "gad7", "pdi", "qids", "cssrs", "lshs", "ahrs"]:
        target_col = f"{target_splitted}_final"
    
    '''
    elif "phq9" in target:
        target_col = "phq9_final"
        
    elif "gad7" in target:
        target_col = "gad7_final"

    elif ""
    '''
                        #for QIDS, CSSRS, PDI, LSHS, AHRS
    if ps == 1:
        X, y, features = get_dataset_redcap_ps1(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv)

                      #for QIDS, CSSRS, PDI, LSHS, AHRS 
    elif ps == 1.5:
        X,y, features = get_dataset_redcap_ps1_prime(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv)

                       #for QIDS, CSSRS, PDI, LSHS, AHRS
    elif ps == 1.75:
        X,y, features = get_dataset_redcap_ps1_prime_2(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv)

    elif ps == 2:
        X,y, features = get_dataset_ema_ps2(target, window_size, is_sliding, ema_csv, redcap_id)
    elif ps == 2.5: #ps == 2.5
        X,y, features = get_dataset_redcap_ps2_prime(target, target_col, window_size, is_sliding, redcap_id)

    elif ps == 3:
        X,y, features = get_dataset_ema_ps3(target, window_size, is_sliding, participant_id, app_act_csv, ema_csv)

    elif ps == 3.5:
        X,y, features = get_dataset_redcap_ps3_prime(target, target_col, window_size, is_sliding, redcap_id, app_act_csv)

    elif ps == 4:
        X,y, features = get_dataset_ema_ps4(target, window_size, is_sliding, ema_csv, participant_id)

    elif ps == 4.5:
        X,y, features = get_dataset_redcap_ps4_prime(target, target_col, window_size, is_sliding, redcap_id)

    elif ps == 5:
        X,y, features = get_dataset_ema_ps5(target, window_size, is_sliding, ema_csv, participant_id, screen_time_csv)

    elif ps == 5.5:
        X,y, features = get_dataset_redcap_ps5_prime(target, target_col, window_size, is_sliding, redcap_id, screen_time_csv)

    elif ps ==6:
        X,y, features = get_dataset_redcap_ps6(target, target_col, window_size, is_sliding, redcap_id, light_temp_csv)
    #agg ps
    elif ps == 7:
        X,y, features = get_dataset_ema_ps7(target, window_size, is_sliding, participant_id, sleep_act_features_csv, light_temp_csv, ema_csv)

    elif ps == 7.5:
        X,y, features = get_dataset_redcap_ps7_prime(target, target_col, window_size, is_sliding, redcap_id, sleep_act_features_csv, light_temp_csv)

    elif ps == 8:
        X,y, features = get_dataset_ema_ps8(target, window_size, is_sliding, participant_id, app_act_csv, call_csv, screen_time_csv, ema_csv)

    elif ps == 8.5:
        X,y, features = get_dataset_redcap_ps8_prime(target, target_col, window_size, is_sliding, participant_id, redcap_id, app_act_csv, call_csv, screen_time_csv)

    elif ps == 9:
        X,y, features = get_dataset_ema_ps9(target, window_size, is_sliding, participant_id, app_act_csv, ema_csv)

    elif ps == 9.5:
        X,y, features =  get_dataset_redcap_ps9_prime(target, target_col, window_size, is_sliding, participant_id, redcap_id, app_act_csv)

    elif ps == 10:
        X,y, features = get_dataset_ema_ps10(target, window_size, is_sliding, participant_id, sleep_act_features_csv, light_temp_csv, app_act_csv, ema_csv)

    elif ps == 10.5:
        X,y, features = get_dataset_ema_ps10_prime(target, target_col, window_size, is_sliding, participant_id, redcap_id, sleep_act_features_csv, light_temp_csv, app_act_csv)

    return X, y, features


################################ only for regression #####################################
                                                    #relative path to where to store plot, including file name
def cross_val_regression(k, X, y, features, model):
    kf = KFold(n_splits=k, shuffle=True)
    mses = []
    r_squares = []
    shaps_per_fold = [] #fold_num -> shap values for said test fold
    train_errors = []
    test_indices = []
    

    if X.shape[0] < 5:
        return -1,[], -1, -1

    for (train_idx, test_idx) in(kf.split(X,y)):
        X_train, y_train = X[train_idx,:], y[train_idx]
        X_test, y_test = X[test_idx,:], y[test_idx]
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        ##print("y_train", mean_squared_error(y_train, y_train_pred))
        ##print("y_test", mean_squared_error(y_test, y_pred))

        mses.append(mean_squared_error(y_test, y_pred)) #test error
        r_squares.append(r2_score(y_test, y_pred))
        train_errors.append(mean_squared_error(y_train, y_train_pred))

        df_X_test = pd.DataFrame(columns=features, data=X_test)
        df_X_test = df_X_test.astype(float)
        explainer = shap.TreeExplainer(model) #shap for each fold
        
        try:
            shap_values = explainer(df_X_test)

        ##print("X_test", X_test)
        ##print("shap values", shap_values.shape)
        ##print(type(shap_values))
        ##print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            for s in shap_values:
                ##print("individual s", s.shape)
                shaps_per_fold.append(s)
             #index of datapoint corresponding to the above shap values

            for index in test_idx:
                test_indices.append(index)
        except ValueError:
            print("can't compute shap")

        
    v = np.array(list(map(lambda x : x.values, shaps_per_fold)))
    datapoints =  np.array(list(map(lambda x : x.data, shaps_per_fold)))
    if v.size != 0:
        all_shaps = shap.Explanation(values=v, data=datapoints)
        all_shaps.feature_names = features

    else:
        all_shaps = []
    
    ##print("all shap shape", shaps.shape)
    
    return np.nanmean(mses), np.nanmean(r_squares), all_shaps, test_indices, np.nanmean(train_errors)
                                                                            #folder name        file name
def avg_cross_val_regression(k, X, y, features, model, model_max_depth=5, shap_plot_loc="", shap_plot_name=""):
    if model == "rf":
        model = RandomForestRegressor(model_max_depth)
    elif model == "xgboost":
        model = xgb.XGBRegressor(objective='reg:squarederror')
    mses = []
    r_squares = []
    shaps_values_per_run = []
    train_errors = []
    test_indices_per_run = []
    
    if not os.path.isdir(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}"):
        #print("all model configs created folder")
        os.mkdir(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}")

    #if len(os.listdir(f"/data/benlab/users/dylan_saiyara/shap_plots/{shap_plot_loc}")) != 0: #ran that config
        #return "already done", "already done",
    
    could_compute_shap = True

    for i in range(10):
        mse, r_square, s, test_indices, train_error = cross_val_regression(k, X, y, features, model)
        mses.append(mse)
        r_squares.append(r_square)
        shaps_values_per_run.append(s)
        if s == []:
            could_compute_shap = False
        
        train_errors.append(train_error)
        test_indices_per_run.append(test_indices)

    if could_compute_shap:
    
        datapoints_per_run = np.array(list(map(lambda x : x.data, shaps_values_per_run))) #permutation of datapoints in test set per rin
        values_per_run = np.array(list(map(lambda x : x.values, shaps_values_per_run))) # now average for each datapoint across all runs
        ##print(datapoints_per_run.shape)
        ##print(values_per_run.shape)

        
        datapoint_to_shaps = {} #map each datapoint to shap values across all runs

        for run in range(datapoints_per_run.shape[0]):
            test_indices_for_run = test_indices_per_run[run]
            values_for_run = values_per_run[run]
            for index in test_indices_for_run:
                if index not in datapoint_to_shaps:
                    datapoint_to_shaps[index] = []
                datapoint_to_shaps[index].append(values_for_run[index])


        def avg_shap_for_datapoint(v):
            return np.nanmean(v, axis=0)

        datapoints = []
        values = []

        for k, v in datapoint_to_shaps.items():
            datapoints.append(X[k])
            values.append(avg_shap_for_datapoint(v))

        all_shaps = shap.Explanation(values=np.array(values), data=np.array(datapoints))
        all_shaps.feature_names = features

        
        shap.plots.beeswarm(all_shaps, show=False)
            
        pl.savefig(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}/{shap_plot_name}.png", bbox_inches='tight')
        pl.close()

    else:
        all_shaps = []

    return np.nanmean(mses), np.nanstd(mses), np.nanmean(r_squares), np.nanstd(r_squares), all_shaps, np.nanmean(np.array(train_errors)), f"{shap_plot_loc}/{shap_plot_name}.png"

################################ only for classification #########################################

                                                        #relative path to where to store plot, including file name
def cross_val_classification(k, X, y, features, model): 
    
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    accs =[]
    aucs = []
    f1s = []
    sensitivities = []
    specificities = []
    npvs = []
    ppvs = []

    shap_values_per_fold = []
    test_indices =  []

    train_errors = []

    
    num_classes = np.unique(y).shape[0]  #num classes in y is guaranteed >= 2 because I check label dist before running
   
    if num_classes == 2:
        pred_type = "binary"
    else:
        pred_type = "multi"

    if X.shape[0] < 5:
        return -1, -1, -1, -1, -1, -1, []
    changing_classes =False
    original_classes = None
    
    if not np.isin(0, y): #xgboost needs classes to start from 0
        #change the first class to 0 and the rest follow prior to training
        changing_classes =  True
        first_class = int(np.min(y))
        to_subtract = first_class
        original_classes = np.unique(y)

        for i in range(y.shape[0]):
            y[i] -= to_subtract
    
        #then modify shap accordingly
    
    try:  
        
        for (train_idx, test_idx) in (kf.split(X,y)):
            X_train, y_train = X[train_idx,:], y[train_idx]
            X_test, y_test = X[test_idx,:], y[test_idx]
            
            model.fit(X_train, y_train)
            ##print("model classes", model.classes_)

            y_train_pred = model.predict(X_train)
            
            y_pred = model.predict(X_test)


            y_pred_prob = model.predict_proba(X_test)

            ##print("y test", y_test, y_test.shape)
            ##print("y pred", y_pred, y_pred.shape)
            ##print("y_pred_prob", y_pred_prob.shape)
            accs.append(accuracy_score(y_test, y_pred))
            train_errors.append(accuracy_score(y_train, y_train_pred))
            try:
                if pred_type == "binary":
                    aucs.append(roc_auc_score(y_test, y_pred, average = "micro", multi_class='ovr')) 
                           
                else:
                    aucs.append(roc_auc_score(y_test, y_pred_prob, average = "micro", multi_class='ovr'))
                    
                
                f1s.append(f1_score(y_test, y_pred, average = "micro"))

            except ValueError:
                print("Cant compute AUC and/or F1")
                aucs.append(np.nan)
                f1s.append(np.nan)
            if pred_type == "binary":
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist() 
            else:
                cm = confusion_matrix(y_test, y_pred)
                fp = cm.sum(axis=0) - np.diag(cm)  
                fn = cm.sum(axis=1) - np.diag(cm)
                tp = np.diag(cm)
                tn = cm.sum() - (fp + fn + tp)

            try: #averaging over all class in multiclass
                if pred_type == "binary":
                    sensitivities.append((tp/(tp+fn)))
                else:
                    sensitivities.append(np.nanmean(tp/(tp+fn)))
                
            except :
                print("cant compute sensitivity")
                sensitivities.append(np.nan)
            
            try:
                if pred_type == "binary":
                    specificities.append((tn / (tn + fp)))
                else:
                    specificities.append(np.nanmean(tn / (tn + fp)))
                
            except :
                print("cant compute specificity")
                specificities.append(np.nan)
            
            try:
                if pred_type == "binary":
                    npvs.append((tn/(tn+fn)))
                else:
                    npvs.append(np.nanmean(tn/(tn+fn)))

            except:
                print("cant compute npv")
                npvs.append(np.nan)
            try:
                if pred_type == "binary":
                    ppvs.append((tp/(tp+fp)))
                else:
                    ppvs.append(np.nanmean(tp/(tp+fp)))
            
            except:
                print("cant compute ppv")
                ppvs.append(np.nan)


            df_X_test = pd.DataFrame(columns=features, data=X_test)
            df_X_test = df_X_test.astype(float)
            explainer = shap.TreeExplainer(model)
            
            
            try:
                shap_values = explainer(df_X_test) #shape of shap depends of dist of y_pred
                                                #if only one class is always predicted the shape is (datapoints, features),
                                                # else it's (datapoints, features, num classes in y_pred)

                                                
                                                # config in automated_model_running

                                                #cross val in such a way so that maintain shape of shap values to match num classes
                
                if changing_classes:#should also account for requirement to shift classes and every cross validation fold not having all classes
                    new_shap_values = np.full((shap_values.shape[0], shap_values.shape[1], np.max(original_classes)+1), np.nan)
                    
                    for changed_class in np.unique(y):
                        original_class = changed_class + to_subtract
                        
                        new_shap_values[:, :, original_class] = shap_values.values[:, :, changed_class]
                        shap_values.values = new_shap_values
                        
                    #print("0\n",new_shap_values[:, :, 0])
                    #print("1\n",new_shap_values[:, :, 1])
                    #print("2\n",new_shap_values[:, :, 2])
                    #print("3\n",new_shap_values[:, :, 3])
                else:
                
                    if model.classes_.shape[0] < np.unique(y).shape[0]: #check if any class is missing in this cross validation fold
                        for c in np.unique(y):
                            if c not in model.classes_: #for that class add np.null just to make sure shap values are always the same shape across cross val folds
                                
                                new_values = np.insert(shap_values.values, c, np.full((shap_values.shape[0], shap_values.shape[1]), np.nan), axis=2)
                                shap_values.values = new_values
                #print("shap shape", shap_values.shape)
                #print("X test shape", X_test.shape)
                #print("y_pred", y_pred)
                #print("y test", y_test)
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                ##print("X test", X_test.shape)
                ##print("y pred", y_pred)
                ##print("shap", shap_values.shape)

                ##print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                ##print("all shap values", shap_values.shape)
                for s in shap_values: #shap values for each datapoint in test fold
                    ##print("individual shap", s.shape)
                    shap_values_per_fold.append(s)

                for index in test_idx:
                    test_indices.append(index)
            except ValueError:
                print("can't compute shap")

        v = np.array(list(map(lambda x : x.values, shap_values_per_fold)))
        datapoints =  np.array(list(map(lambda x : x.data, shap_values_per_fold)))
        if v.size != 0:
            all_shaps = shap.Explanation(values=v, data=datapoints)
            all_shaps.feature_names = features  
        else:
            all_shaps = []

        ##print(datapoints.shape)
        ##print(v.shape)
        ##print(datapoints.shape)
        ##print(all_shaps.shape)
        '''
        if len(all_shaps.shape) == 2:
            shap.plots.beeswarm(all_shaps) #xgboost doesn't have one extra dim for binary preds
        else:
            shap.plots.beeswarm(all_shaps[:, :, 0])#plot for one class only
        '''
        #pl.savefig(f"/data/benlab/users/dylan_saiyara/shap_plots/{shap_plot_loc}.png", bbox_inches='tight')
        #pl.close()
    
    except ValueError: #not enough for cross validation split
        return -1, -1, -1, -1, -1, -1, -1, [], -1, -1, None
    
    return np.nanmean(accs), np.nanmean(aucs), np.nanmean(f1s), np.nanmean(sensitivities), np.nanmean(specificities), np.nanmean(npvs), np.nanmean(ppvs), all_shaps, test_indices, np.nanmean(train_errors), pred_type

                               #check dist of y labels

def train_test_rf(X,y, features, criterion, k=5, model_max_depth=5):
    
    y = np.array(list(map(criterion, y)))
    
    model = RandomForestClassifier(max_depth=model_max_depth)
    
    return cross_val_classification(k, X, y, features, model)

def train_test_xgboost(X, y, features, criterion, k=5):
    y = np.array(list(map(criterion, y)))
    num_classes = np.unique(y).shape[0]


    if num_classes == 2:
        model = xgb.XGBClassifier(objective="binary:logistic")
    else:
        model = xgb.XGBClassifier(objective="multi:softprob")

    
    return cross_val_classification(k, X, y, features, model)

### no need to compute features and target means over and over again,
### just compute X and y once and run several times and average
                                                                            #folder name        file name
def avg_cross_val_auc_rf(X,y, features, criterion, k=5, model_max_depth=5, shap_plot_loc="", shap_plot_name=""):
    accs= []
    aucs = []
    f1s = []
    sensitivities = []
    specificities = []
    npvs = []
    ppvs = []

    shaps_values_per_run = []
    train_errors = []
    test_indices_per_run = []
    
   
    if not os.path.isdir(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}"):
        #print("all model configs created folder")
        os.mkdir(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}")
    
    #if len(os.listdir(f"/data/benlab/users/dylan_saiyara/shap_plots/{shap_plot_loc}")) != 0: #ran that config
        #return "already done", "already done", "already done", "already done", "already done", "already done","already done"
    
    could_compute_shap = True
    
    for i in range(10):
        ##print("run", i)
        acc, auc, f1, sensitivity, specificity, npv, ppv, shaps, test_indices, train_error, pred_type = train_test_rf(X,y, features, criterion, k, model_max_depth)
       
        if not np.isnan(acc):
            accs.append(acc)
            ##print(shaps.shape)
            if shaps == []:
                could_compute_shap = False

            shaps_values_per_run.append(shaps)
            test_indices_per_run.append(test_indices)
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(f1):
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            npvs.append(npv)
            ppvs.append(ppv)
            f1s.append(f1)

        if not np.isnan(train_error):
            train_errors.append(train_error)

    if could_compute_shap:
        datapoints_per_run = np.array(list(map(lambda x : x.data, shaps_values_per_run))) #permutation of datapoints in test set per rin
        values_per_run = np.array(list(map(lambda x : x.values, shaps_values_per_run))) # now average for each datapoint across all runs
        ##print(datapoints_per_run.shape)
        ##print(values_per_run.shape)

        
        datapoint_to_shaps = {} #map each datapoint to shap values across all runs

        for run in range(datapoints_per_run.shape[0]):
            test_indices_for_run = test_indices_per_run[run]
            values_for_run = values_per_run[run]
            for index in test_indices_for_run:
                if index not in datapoint_to_shaps:
                    datapoint_to_shaps[index] = []
                datapoint_to_shaps[index].append(values_for_run[index])


        
        def avg_shap_for_datapoint(v):
            return np.nanmean(v, axis=0)

        datapoints = []
        values = []

        for k, v in datapoint_to_shaps.items():
            datapoints.append(X[k])
            
            values.append(avg_shap_for_datapoint(v))
            ##print(k, len(v), avg_shap_for_datapoint(v).shape)
        
        
        all_shaps = shap.Explanation(values=np.array(values), data=np.array(datapoints))
        all_shaps.feature_names = features
        #print(all_shaps.shape)

        #plot for all classes
        for i in range(all_shaps.shape[2]):
            shap.plots.beeswarm(all_shaps[:, : , i], show=False) #for 1 class at a timr, rf model shaps guaranteed to have third dim for both binary and multiclass classification
            pl.savefig(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}/{shap_plot_name}_class_{i}.png", bbox_inches='tight')

            pl.close()

    else:
        all_shaps = []
    
    return np.nanmean(accs), np.nanstd(accs), np.nanmean(aucs), np.nanstd(aucs), np.nanmean(f1s), np.nanstd(f1s), np.nanmean(sensitivities), np.nanstd(sensitivities), np.nanmean(specificities), np.nanstd(specificities), np.nanmean(npvs), np.nanstd(npvs), np.nanmean(ppvs), np.nanstd(ppvs), all_shaps, np.nanmean(np.array(train_errors)), pred_type, f"{shap_plot_loc}/{shap_plot_name}.png"
   #np.nanmean(accs), np.nanmean(aucs), np.nanmean(f1s), np.mean(sensitivities), np.mean(specificities), np.mean(npvs), np.mean(ppvs), all_shaps, test_indices, np.mean(train_errors), pred_type
                                                                        #folder name        file name
def avg_cross_val_auc_xgboost(X,y, features, criterion, k=5, shap_plot_loc="", shap_plot_name=""):
    accs= []
    aucs = []
    f1s = []
    sensitivities = []
    specificities = []
    npvs = []
    ppvs = []
    shaps_values_per_run = []
    train_errors = []
    test_indices_per_run = []

       
    if not os.path.isdir(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}"):
        #print("all model configs created folder")
        os.mkdir(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}")

    #if len(os.listdir(f"/data/benlab/users/dylan_saiyara/shap_plots/{shap_plot_loc}")) != 0: #ran that config
        #return "already done", "already done", "already done", "already done", "already done", "already done","already done"
    
    could_compute_shap = True

    for i in range(10):
        acc, auc, f1, sensitivity, specificity, npv, ppv, shaps, test_indices, train_error, pred_type = train_test_xgboost(X, y, features, criterion, k)
        if not np.isnan(acc):
            accs.append(acc)

            if shaps == []:
                could_compute_shap = False

            shaps_values_per_run.append(shaps)
            test_indices_per_run.append(test_indices)
        if not np.isnan(auc):
            aucs.append(auc)
        if not np.isnan(f1):
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            npvs.append(npv)
            ppvs.append(ppv)
            f1s.append(f1)

        if not np.isnan(train_error):
            train_errors.append(train_error)

    if could_compute_shap:
        datapoints_per_run = np.array(list(map(lambda x : x.data, shaps_values_per_run))) #permutation of datapoints in test set per rin
        values_per_run = np.array(list(map(lambda x : x.values, shaps_values_per_run))) # now average for each datapoint across all runs
            ##print(datapoints_per_run.shape)
            ##print(values_per_run.shape)
        
            
        datapoint_to_shaps = {} #map each datapoint to shap values across all runs
        
        for run in range(datapoints_per_run.shape[0]):
            test_indices_for_run = test_indices_per_run[run]
            values_for_run = values_per_run[run]
            for index in test_indices_for_run:
                if index not in datapoint_to_shaps:
                    datapoint_to_shaps[index] = []
                datapoint_to_shaps[index].append(values_for_run[index])
        
            
        def avg_shap_for_datapoint(v):
            return np.nanmean(v, axis=0)
        
        datapoints = []
        values = []
        
        for k, v in datapoint_to_shaps.items():
            datapoints.append(X[k])
                
            values.append(avg_shap_for_datapoint(v))
            #print(k, len(v), avg_shap_for_datapoint(v).shape)
            
        all_shaps = shap.Explanation(values=np.array(values), data=np.array(datapoints))
        all_shaps.feature_names = features
        #print("all shaps shape", all_shaps.shape)
        
        if len(all_shaps.shape) > 2:
            for i in range(all_shaps.shape[2]):
                shap.plots.beeswarm(all_shaps[:, : , i], show=False) #for 1 class at a time, xgboost for multiclass prediction
                pl.savefig(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}/{shap_plot_name}_class_{i}.png", bbox_inches='tight')
                pl.close()
        else:
            shap.plots.beeswarm(all_shaps, show=False) #xgboost for binary pred
            pl.savefig(f"/data/benlab/users/alice_saiyara/shap_plots/{shap_plot_loc}/{shap_plot_name}_class_0.png", bbox_inches='tight')

            pl.close()
    else:
        all_shaps = []
        
    return np.nanmean(accs), np.nanstd(accs), np.nanmean(aucs), np.nanstd(aucs), np.nanmean(f1s), np.nanstd(f1s), np.nanmean(sensitivities), np.nanstd(sensitivities), np.nanmean(specificities), np.nanstd(specificities), np.nanmean(npvs), np.nanstd(npvs), np.nanmean(ppvs), np.nanstd(ppvs), all_shaps, np.nanmean(np.array(train_errors)), pred_type, f"{shap_plot_loc}/{shap_plot_name}.png"

######################################################################################

