################################# IMPORTS #################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
import copy
import math


############################## HELPER FUNCTIONS ##############################

def weekNumber(df):
    draft_week_number = []
    count = len(df["date"])

    for i in range(0, count):
        date =  df["date"][i]
        start = ((date.month - 1) * 4) + 1
        day = date.day
        if(day >= 1 and day <= 7):
            draft_week_number.append(start)
        if(day >= 8 and day <= 15):    
            draft_week_number.append(start+1)
        if(day >= 16 and day <= 23):    
            draft_week_number.append(start+2)
        if(day >= 24):    
            draft_week_number.append(start+3)

    return draft_week_number

    
def drop_null_cols(df):

    df_in = copy.deepcopy(df)

    for col in df_in.columns:
        if(df_in[col].isnull().sum() == df_in.shape[0] or (len(df_in[col].value_counts())==1 and (df_in[col].value_counts().index[0] == 'Data missing or out of range') or df_in[col].value_counts().index[0] == -1)):
            df_in.drop([col], axis=1, inplace=True)

    return df_in        


# Replacing column with less 2% null values with mode. ( handling MCAR ) 
def replace_with_mode(df, value_to_be_replaced):
    df_in = copy.deepcopy(df)

    for c in df_in.columns:
        if( df_in[c].isin([value_to_be_replaced]).sum()>0 ):

            # is missing value frequency less than thres    
            if((len(df_in[df_in[c]==value_to_be_replaced])/len(df_in))*100 <= 2.0):
                variable_mode = df_in[c].mode()[0]
                df_in[c].replace([value_to_be_replaced],variable_mode, True)

    return df_in


def handle_no_junction(df_in, feature_to_be_modified, feature_for_imputing, value_to_compare, replacement ):
    df_in[feature_to_be_modified].mask(df_in[feature_for_imputing] == value_to_compare, replacement, inplace=True)
    return df_in



# how many to assign top what you want ? 
def calculate_top_categories(df, variable, how_many):
    return [
        x for x in df[variable].value_counts().sort_values(
            ascending=False).head(how_many).index
    ]

# take the df, the feature and the top labels (manually)
def one_hot_encode(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable + '_' + label] = np.where(
            df[variable] == label, 1, 0) 



def number_encode_features(df_in,column):
    df_in[column] = preprocessing.LabelEncoder().fit_transform(df_in[column])   



#fixing local authority district

def dividing_mixed_col(df_in, colname_to_divide, col1_name, col2_name, default_col1, default_col2):

    df_in[col1_name]= np.where(   
    (df_in[colname_to_divide].str.isnumeric()==True) | (df_in[colname_to_divide].str.isnumeric().isnull())
    , df_in[colname_to_divide], default_col1 ).astype(str)

    
    df_in[col2_name]= np.where(   
    (df_in[colname_to_divide].str.isnumeric()==True) | (df_in[colname_to_divide].str.isnumeric().isnull())
    , default_col2, df_in[colname_to_divide] ).astype(str)
           
                

             
############################## ETL FUNCTIONS (M1) ##############################

# load data
def read_data(finalename):
    df = pd.read_csv(finalename, index_col='accident_index')
    return df

def drop_duplicates(df):
    df_no_duplicates = df.drop_duplicates(subset=df.columns.difference(['accident_reference']))
    return df_no_duplicates;

def add_features(df):

    df_new_features = copy.deepcopy(df)

    df_new_features['date'] =  pd.to_datetime(df_new_features['date'], format='%d/%m/%Y')
    # extracting the day of the month 
    df_new_features['drv_day_number'] = pd.DatetimeIndex(df_new_features['date']).day
    # extracting the month number 
    df_new_features['drv_month_number'] = pd.DatetimeIndex(df_new_features['date']).month
    # extracting the month name 
    df_new_features['drv_month_name'] = df_new_features['drv_month_number'].apply(lambda x: calendar.month_abbr[x])
    # extracing hour time 
    df_new_features['hour']=df_new_features['time'].astype('datetime64[ns]').dt.hour
    # extract week number
    week_number = weekNumber(df_new_features)
    df_new_features['week_number'] = week_number

    return df_new_features


def scale_features(df):

    df_scaled_features = copy.deepcopy(df)

    number_of_casualties_original = df.number_of_casualties 
    number_of_casualties_scaled = MinMaxScaler().fit_transform(df[["number_of_casualties"]]) 
    df_scaled_features["number_of_casualties"] = number_of_casualties_scaled

    number_of_vehicles_original = df.number_of_vehicles 
    number_of_vehicles_scaled = MinMaxScaler().fit_transform(df[["number_of_vehicles"]]) 
    df_scaled_features["number_of_vehicles"] = number_of_vehicles_scaled

    return df_scaled_features
    
def handle_missing(df):
    df_in = drop_null_cols(df)

    replace_with_zero = 'first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero '

    df_in.replace({ "first_road_number": {replace_with_zero: 0.0 }}, inplace = True)
    df_in.replace({ "second_road_number": {replace_with_zero: 0.0 }}, inplace = True)

    df_in["first_road_number"].fillna( value=-1, inplace = True)
    df_in["second_road_number"].fillna( value=-1, inplace = True)

    df_in = handle_no_junction(df_in, 'junction_control','junction_detail', 'Not at junction or within 20 metres','No junction')
    df_in = handle_no_junction(df_in, 'second_road_class','junction_detail', 'Not at junction or within 20 metres','None')


    modeOfSecondRoadClassBasedOnJunctionDetail = df_in.groupby(['junction_detail'])['second_road_class'].agg(lambda x: pd.Series.mode(x)[0])
    second_road_class_dict = modeOfSecondRoadClassBasedOnJunctionDetail.to_dict()
    df_secondRoad_temp = df_in['junction_detail'].map(second_road_class_dict)
    df_in['second_road_class'] = np.where(df_in['second_road_class']=='-1', df_secondRoad_temp, df_in['second_road_class'])

    modeRoadType = df_in.groupby(['first_road_class'])['road_type'].agg(lambda x: pd.Series.mode(x)[0])
    roadType_dict = modeRoadType.to_dict()
    df_roadType_temp = df_in['first_road_class'].map(roadType_dict)
    df_in['road_type'] = np.where(df_in['road_type'].isnull(), df_roadType_temp, df_in['road_type'])
    
    # Drop the rows where we have location_easting_osgr and location_northing_osgr as null values because this is less than 0.1% of the data (MCAR)
    df_in.dropna(subset=['location_easting_osgr', 'location_northing_osgr'], inplace= True)


    modeWeather = df_in.groupby(['drv_month_name'])['weather_conditions'].agg(lambda x: pd.Series.mode(x)[0])
    modeWeather_dict = modeWeather.to_dict()
    df_modeWeather_temp = df_in['drv_month_name'].map(modeWeather_dict)
    df_in['weather_conditions'] = np.where(df_in['weather_conditions'].isnull(), df_modeWeather_temp, df_in['weather_conditions'])

    missing_light= df_in[df_in['light_conditions'].str.contains('missing')]
    missing_light['light_conditions'] = np.where((missing_light['hour']>=6) & (missing_light['hour']<17), 'Daylight', 'Darkness')
    df_in.update(missing_light)


    df_in = replace_with_mode(df_in, 'Data missing or out of range')
    df_in = replace_with_mode(df_in, '-1')

    print('handling missing successful')

    return df_in

def remove_outliers(df_M):

    # drop rows on the right of the boundry only if the % is < 1%
    if((len(df_M[df_M['location_easting_osgr']> 660000])/len(df_M))*100 < 1 ):
        df_M.drop( df_M[df_M['location_easting_osgr']> 660000].index, inplace = True)   
    return df_M     



def transfom_features(df):

    df_T = copy.deepcopy(df)
    accident_everity_encoded = df_T.replace({'accident_severity': {'Slight': 0, 'Serious': 1, 'Fatal': 2}})
    df_T['accident_severity'] = accident_everity_encoded['accident_severity']


    df_T["has_pedestrian_crossing_human_control"] = np.where(df_T['pedestrian_crossing_human_control'].str.contains('None'), 0, 1)
    df_T["has_pedestrian_crossing_physical_facilities"] = np.where(df_T['pedestrian_crossing_physical_facilities'].str.contains('No'), 0, 1)


    df_T['light_conditions'] = np.where(df_T['light_conditions'].str.contains('lights lit'), 'Lighting', df_T['light_conditions'])
    df_T['light_conditions'] = np.where(df_T['light_conditions'].str.contains('Darkness'), 'Darkness', df_T['light_conditions'])
    light_contitions_encoding = { "light_conditions": {"Daylight": 2, "Lighting": 1, "Darkness": 0 }}
    df_T.replace(light_contitions_encoding, inplace=True)


    df_T['weather_wind_intensity'] = np.where(((df_T['weather_conditions'].str.contains('no high')) | (df_T['weather_conditions'].str.contains('wind')==False)) , 0, 1)
    df_T['weather_is_fine'] = np.where(((df_T['weather_conditions'].str.contains('Fine'))) , 1, 0)
    df_T['weather_is_other'] = np.where(((df_T['weather_conditions'].str.contains('Other'))) , 1, 0)


    df_T['road_surface_conditions'] = np.where(((df_T['road_surface_conditions'].str.contains('Wet')) | (df_T['road_surface_conditions'].str.contains('Dry'))) , df_T['road_surface_conditions'], 'Other')
    road_surface_categories = df_T['road_surface_conditions'].value_counts().index.tolist()
    one_hot_encode(df_T,'road_surface_conditions',road_surface_categories)


    df_T['has_special_conditions_at_site'] = np.where( df_T['special_conditions_at_site'] == 'None', 0, 1)  # 0:None 1:Exists


    df_T['has_carriageway_hazards'] = np.where(df_T['carriageway_hazards'] == 'None', 0, 1)  # 0:None 1:Exists


    df_T['junction_control'] = np.where(((df_T['junction_control'].str.contains('uncontrolled')) ) , 1, df_T['junction_control'])  # 1:uncontrolled 
    df_T['junction_control'] = np.where( ~(df_T['junction_control']==1) & (df_T['junction_control'].str.contains('No junction')) , 0, df_T['junction_control'])  # 0: no junction
    df_T['junction_control'] = np.where( ~((df_T['junction_control']==0)) & ~(df_T['junction_control']==1) , 2, df_T['junction_control'])  # 2: Exists some form of junction


    road_type_categories = df_T['road_type'].value_counts().index.tolist()
    one_hot_encode(df_T,'road_type',road_type_categories)


    df_T['junction_detail'] = np.where(((df_T['junction_detail'].str.contains(df_T['junction_detail'].value_counts().index[0])) | (df_T['junction_detail'].str.contains(df_T['junction_detail'].value_counts().index[1]))) , df_T['junction_detail'], 'Other')  # 0:uncontrolled or not a junction   1: Exists a form of junction control
    junction_detail_categories = df_T['junction_detail'].value_counts().index.tolist()
    one_hot_encode(df_T,'junction_detail',junction_detail_categories)


    road_class_encoding = { "first_road_class": {"A": 4, "B": 3, "C": 2, "Unclassified": 1, "Motorway": 5, "A(M)":5 }, "second_road_class": {"A": 4, "B": 3, "C": 2, "Unclassified": 1, "Motorway": 5, "A(M)":5, "None": 0 }}
    df_T.replace(road_class_encoding, inplace=True)
    df_T['first_road_class'] =  df_T['first_road_class'].astype(int)
    df_T['sencond_road_class'] =  df_T['second_road_class'].astype(int)


    df_T['speed_limit']=np.where( df_T['speed_limit']<10, 10, df_T['speed_limit'])
    df_T['speed_limit']=np.where(df_T['speed_limit'] % 10>=5, df_T['speed_limit']+(10-df_T['speed_limit'] % 10),df_T['speed_limit']-(df_T['speed_limit'] % 10)  )


    dividing_mixed_col(df_T, 'local_authority_district', "local_authority_district_code", "local_authority_district_name", 0,  "No Name Recorded" )


    number_encode_features(df_T, 'local_authority_district_name')
    df_T['local_authority_district_name'] = np.where(df_T['local_authority_district_name'] == 0, -1, df_T['local_authority_district_name'] )
    df_T['local_authority_district_name'] = np.where(df_T['local_authority_district_name'] == 184, 0, df_T['local_authority_district_name'] )
    df_T['local_authority_district_name'] = np.where(df_T['local_authority_district_name'] == -1, 184, df_T['local_authority_district_name'] )


    number_encode_features(df_T, 'police_force')


    day_of_week_encoding = { "day_of_week": {"Friday": 4, "Saturday": 5, "Sunday": 6, "Monday": 0, "Tuesday":1, "Wednesday":2, "Thursday":3 }}
    df_T.replace(day_of_week_encoding, inplace=True)

    print('transformation successful')

    return df_T


def drop_redundant(df_in):
    df_in.drop(['local_authority_district','road_type','junction_detail',
            'carriageway_hazards','pedestrian_crossing_physical_facilities','pedestrian_crossing_human_control',
            'weather_conditions','special_conditions_at_site','road_surface_conditions','drv_month_name','time','date'], axis=1, inplace=True)
    
    df_in.drop(['accident_year','accident_reference'], axis=1, inplace=True)


############################## NEW FEATURES M2 ##############################


def fill_missing_with_external(df_in):

    df = copy.deepcopy(df_in)
    # our encoding
    la_encoding= pd.read_csv('external_data\LA_encoding.csv').set_index('code')

    #external resource
    LAD_codes_names = pd.read_csv('external_data\Local_authority_and_regional_breakdown_January_2019.csv').set_index('Local authority')
    lad_code_name_dict = LAD_codes_names.to_dict().get('Unnamed: 1')

    # getting_names from encoding
    la_encoding.name = la_encoding.name.str.strip()
    la_dict = la_encoding.to_dict().get('name')

    # convert encodings to dictionary key: name value: code
    la_encoding_reversed = la_encoding.copy()
    la_encoding_reversed = la_encoding_reversed.reset_index()
    la_encoding_reversed = la_encoding_reversed.set_index('name')
    la_dict_reversed = la_encoding_reversed.to_dict().get('code')
    
    df = df.reset_index()

    df['local_authority_district_name'] = df['local_authority_district_name'].astype('int')
    df['local_authority_district_name']= df['local_authority_district_name'].map(la_dict)

    # remove anything after the comma for example: "city of".
    df['local_authority_district_name'] = df['local_authority_district_name'].str.rsplit(',', n=1).str.get(0)

    #get all districts' codes that has no name recorded
    no_district_name = df[df['local_authority_district_name'].str.contains("No Name Recorded")]
    no_district_name['local_authority_district_code'] = no_district_name['local_authority_district_code'].astype('int')
    # print(len(no_district_name))
    no_district_name['local_authority_district_name'] = no_district_name['local_authority_district_code'].map(lad_code_name_dict)

    no_district_name = no_district_name.set_index('accident_index')
    
    df = df.set_index('accident_index')

    df.update(no_district_name)

    df = df.reset_index()

    ################# isUrban ########################

    df_urban = pd.read_excel('external_data\isurbancodes.xls')

    df_urban = df_urban.drop_duplicates('Urban area name')
    df_FE_urban = pd.merge(left=df, right=df_urban['Urban area name'], left_on='local_authority_district_name', right_on='Urban area name', how='left')
    df_FE_urban['isUrban'] = np.where(df_FE_urban['Urban area name'].isnull(), 0, 1)

    df = df_FE_urban.copy()
    df.drop(['Urban area name'], axis=1, inplace=True)
    # print("in_set", df.accident_index)
    # encode
    df['local_authority_district_name']= df['local_authority_district_name'].map(la_dict_reversed)


    return df


def get_gender(df_in):

    df = copy.deepcopy(df_in)
    
    accidents_index = pd.read_csv("external_data\data_accident_idx.csv")
    facts_veh_df = pd.read_csv('external_data\data_facts_veh.csv')
    vehicle_driver  = pd.read_csv("external_data\data_vehicle_driver.csv")

    merge1 = pd.merge(df, accidents_index, left_on='accident_index', right_on='src_accident_index' )
    merge2 = pd.merge(merge1, facts_veh_df,left_on='drv_road_safety_accident_index_key', right_on='drv_road_safety_accident_index_key')
    merge3 = pd.merge(merge2,vehicle_driver, left_on='drv_road_safety_vehicle_driver_key', right_on='drv_road_safety_vehicle_driver_key')



    merge3['src_sex_of_driver'].replace(['Data missing or out of range'],'Not known', True)
    gender_list = merge3['src_sex_of_driver'].value_counts().index.tolist()
    one_hot_encode(merge3, 'src_sex_of_driver', gender_list)


    df = df.set_index('accident_index')

    males_count = merge3.groupby('src_accident_index').sum()[['src_sex_of_driver_Male']]
    females_count = merge3.groupby('src_accident_index').sum()[['src_sex_of_driver_Female']]

    df['males_count'] = males_count
    df['females_count'] = females_count

    df = df.reset_index()

    return df



############################## PIPELINE ##############################

def clean_transform(df):

    df_new = drop_duplicates(df)
    df_new = add_features(df_new)
    df_new = handle_missing(df_new)
    df_new = remove_outliers(df_new)
    df_new = transfom_features(df_new)

    drop_redundant(df_new)

    return df_new

def extract_additional_rsc(df):

    df_new = fill_missing_with_external(df)
    df_new = get_gender(df_new)

    return df_new

def load_to_csv(df, filename):
    print(df.shape)
    print(df.columns)
    df.to_csv(filename,index=False)
    print('loaded csv after cleaning succesfully')

    
def full_ETL_pipeline():
    
    df = read_data('1985_Accidents_UK.csv')
    df_new = clean_transform(df)
    df_new_ext = extract_additional_rsc(df_new)

    load_to_csv(df_new_ext, '1985_Accidents_UK_cleaned.csv')


full_ETL_pipeline()  



    