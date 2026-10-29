import pandas as pd
import numpy as np
import math
from predictors import *
from features_list import *

################################### window thing #####################################################
#Get all the windows of target and predictor
                                    # redcap questionnaires are now in dataframes with week start, end and score.
                                    #I need the name of the column corresponding to the score column, target col
                                    #target col can be None for ema-based predictions
                                                                                                    #only when checking num datapoints no need to compute GPS features

def find_where_window_point_lies_on_df_target(window_point, df_target):
    index = None
    for i in range(len(df_target)):
        q_start = df_target["start_of_range"].iloc[i]
        q_end = df_target["end_of_range"].iloc[i]
        if q_start <= window_point <= q_end:
            index = i
            break
    return index

def window(df_features, df_target, target_col, window_len, is_sliding):#with both features and targets, right now it's only for [redcap q's, activity/sleep and all (PS1)] and [ema, app data (PS2)] and window of len 1/2 weeks

    df_target = df_target.dropna().copy(deep=True)
    df_features = df_features.dropna()
    #this gets rid of feature and target dates for which there isn't any data
    #however, we can't know when that's the case for screen time as data is queried on the fly coz the dataset is so large
    #i haven't saved processed results unlike say for app activity because features are aggregated within window, like mean at each hour of day within window
    #therefore, if that's the case select_subset() just exists for that range of dates and doesn't add anything to X and y

    X = []
    y = []

    def earlier_date(d1, d2):
        if d1 <= d2:
            return d1
        else:
            return d2
    def later_date(d1, d2):
        if d1 >= d2:
            return d1
        else:
            return d2
        
    def encode_time(timestamp_time):
        l = [timestamp_time.hour, timestamp_time.minute, timestamp_time.second]
        arr = np.array(l, dtype=np.float64)
        arr[1] /= 60
        arr[2] /= 3600

        total_hours = arr[0] + arr[1] + arr[2]

        return total_hours/24
  
    
    df_features_start_date = df_features["date"].iloc[0]
    df_features_end_date = df_features["date"].iloc[-1]

    #for PHQ7 and GAD7 which have range of dates
    #for QIDS, CSSRS, PDI, LSHS, AHRS
    if "start_of_range" in list(df_target.columns):
        df_target_start_date = df_target["start_of_range"].iloc[0]
        df_target_end_date = df_target["end_of_range"].iloc[-1]
    
    #for emas which don't have range of dates
    else:
        df_target_start_date = df_target["date"].iloc[0]
        df_target_end_date = df_target["date"].iloc[-1]

    start_date = later_date(pd.to_datetime(df_features_start_date), pd.to_datetime(df_target_start_date))
    end_date = earlier_date(pd.to_datetime(df_features_end_date), pd.to_datetime(df_target_end_date))

    #For PS2 (app features) calcuted staypoints, triplegs and trips only once on all available GPS data, regardless of window or intersection with target metric df  
    
    

    def select_subset(curr_date, end_of_range):
        
        #print(curr_date, end_of_range)
        #Inclusive of curr_date, exclusive of end_of_range; curr_date to end_of_range - 1d inclusive
        feature_subset = df_features[(curr_date <= df_features["date"]) & (df_features["date"] < end_of_range)].copy(deep=True) #end feature/ema subset right before end_or_range starts
        #print("feature subset len", len(feature_subset))
        #ONLY FOR REDCAP QUESTIONNAIRES
        #iterating through df_target :<
        #find where curr_date and end_of_range - 1 day lie in df_target_portion
        if "start_of_range" in list(df_target.columns):
            target_subset_start_index = None
            target_subset_end_index = None

            for i in range(len(df_target)):
                #PHQ9 and GAD7 scores are valid from week_start to week_end, inclusive
                if "start_of_range" in list(df_target.columns):
                    # for QIDS, CSSRS, PDI, LSHS, AHRS; this wouldn't be week start and end, but like month/fortnight/2 day period start
                    week_start = df_target["start_of_range"].iloc[i]
                    week_end = df_target["end_of_range"].iloc[i]
                
                if week_start <= curr_date <= week_end:
                    target_subset_start_index = i
                
                if week_start <= end_of_range - pd.Timedelta(1, "D") <= week_end:
                    target_subset_end_index = i

                if target_subset_start_index != None and target_subset_end_index != None: #both start and end of window lie inside ranges of dates covered by questionnaires 
                    break 
                
            if target_subset_start_index != None and target_subset_end_index != None: 
                target_subset = df_target[target_subset_start_index : target_subset_end_index+1]

            else: #need to iteratively decrease window range of dates so that the new start and end lie on range of dates covered by q-> edge case
                new_curr_date = curr_date
                new_end_of_range = end_of_range

                #edge case 1: both ends of window don't fall into ranges of q dates
                if target_subset_start_index == None and target_subset_end_index == None:
                    #print("\nNeed to readjust both")
                    while new_curr_date < new_end_of_range - pd.Timedelta(1, "D") and (target_subset_start_index == None and target_subset_end_index == None):
                        new_curr_date += pd.Timedelta(1, "D")
                        new_end_of_range -= pd.Timedelta(1, "D")
                        if target_subset_start_index == None:
                            target_subset_start_index = find_where_window_point_lies_on_df_target(new_curr_date, df_target)
                        if target_subset_end_index == None:
                            target_subset_end_index = find_where_window_point_lies_on_df_target(new_end_of_range - pd.Timedelta(1, "D"), df_target)
                
                #edge case 2: only start of window doesn't fall into ranges of q dates
                elif target_subset_start_index == None and target_subset_end_index != None:
                    #print("\nNeed to readjust start")
                    while new_curr_date < new_end_of_range - pd.Timedelta(1, "D") and target_subset_start_index == None:
                        new_curr_date += pd.Timedelta(1, "D")
                        if target_subset_start_index == None:
                            target_subset_start_index = find_where_window_point_lies_on_df_target(new_curr_date, df_target)

                #edge case 3: only end of window doesn't fall into ranges of q dates
                elif target_subset_start_index != None and target_subset_end_index == None:
                    #print("\nNeed to readjust end")
                    while new_curr_date < new_end_of_range - pd.Timedelta(1, "D") and target_subset_end_index == None:
                        new_end_of_range -= pd.Timedelta(1, "D")
                        if target_subset_end_index == None:
                            target_subset_end_index = find_where_window_point_lies_on_df_target(new_end_of_range - pd.Timedelta(1, "D"), df_target)

                if target_subset_start_index != None and target_subset_end_index != None: 
                    target_subset = df_target[target_subset_start_index : target_subset_end_index+1]

                elif target_subset_start_index != None and target_subset_end_index == None:
                    target_subset = df_target[target_subset_start_index : target_subset_start_index+1]

                elif target_subset_start_index == None and target_subset_end_index != None:
                    target_subset = df_target[target_subset_end_index : target_subset_end_index+1]
                #even after readjusting window couldn't fit start and end into q range of dates
                else:
                    target_subset = pd.DataFrame([]) #no target q completed during this window, even after decreasing size iteratively
            
                #print("adjusted window", new_curr_date, new_end_of_range, target_subset_start_index, target_subset_end_index)
                #print("---------------------------------------------------")          
            
            
        #FOR EMAs
        else:
            target_subset = df_target[(curr_date <= df_target["date"]) & (df_target["date"] < end_of_range)].copy(deep=True)

        

        #print("Features:", len(feature_subset))

        #print("Targets\n", target_subset["date"])
        #print("#########################################################################")

        #use staypoints df when working with GPS data only

        #for PS1 (sleep/activity), PS7 and PS10 only because they have a sleep or rise time feature
        if "Sleep.Onset.Time" in list(feature_subset.columns):
            feature_subset["Sleep.Onset.Time"] = feature_subset["Sleep.Onset.Time"].apply(pd.to_datetime).apply(pd.Timestamp.time).apply(encode_time)
            feature_subset["Rise.Time"] = feature_subset["Rise.Time"].apply(pd.to_datetime).apply(pd.Timestamp.time).apply(encode_time)
            
        if len(feature_subset) != 0 and len(target_subset) != 0:
            #for PS1 (sleep/activity) only because they have a sleep or rise time or steps feature
            #also for PS7 (agg geneactiv)
            #not PS10 (all features)
            #print(feature_subset.columns)
            #print(feature_subset)
            X.append(feature_subset.drop(columns=["date"]).mean().to_numpy())

    
            if "start_of_range" in list(df_target.columns): #for redcap questionnaires:
                y.append(target_subset[target_col].mean())

            else:
                y.append(target_subset.drop(columns=["date"]).mean().to_numpy().sum())
                                                            #average across dates in target_subset, then add across all individual emas, for ema_diff_prev only because i don't total it first
        
        #print(f'Elapsed during subsets: {end_time_s - start_time_s:.2f} seconds')

    def static():
        
        curr_date = start_date
        while curr_date <= end_date:
            
            end_of_range = curr_date + pd.Timedelta(days=window_len)
            select_subset(curr_date, end_of_range)
            #print("start: ", curr_date, "finish: ", end_of_range)
            curr_date = end_of_range
            

    def sliding():
        
        curr_date = start_date
        while curr_date <= end_date:
            end_of_range = curr_date + pd.Timedelta(days=window_len)
                
            select_subset(curr_date, end_of_range)
            #print("start: ", curr_date, "finish: ", end_of_range)
            curr_date = curr_date +pd.Timedelta(days = math.ceil(window_len / 2))
            
        #print(f'Elapsed after subsets: {end - start:.2f} seconds')
    if is_sliding:
        sliding()
    else:
        static()
    
    return np.array(X), np.array(y)

