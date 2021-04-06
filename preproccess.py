import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

target_year = 2013

# map countries to common format
country_mapping = {
    "Dem. Rep. of Congo" : "Congo, Dem. Rep.", 
    "Congo-Brazzaville" : "Congo, Rep.",
    "Swaziland" : "Eswatini",
    "Kyrgyzstan" : "Kyrgyz Republic", 
    "Laos" : "Lao",
    "Korea, North" : "North Korea",
    "Macedonia" : "North Macedonia",
    "Korea, South" : "South Korea",
    "Sudan (North)" : "Sudan",
    "Timor Leste" : "Timor-Leste",
    "Cabo Verde" : "Cape Verde",
    "China (People's Republic of)" : "China",
    "Congo" : "Congo, Rep.",
    "Côte d'Ivoire" : "Cote d'Ivoire",
    "Democratic People's Republic of Korea" : "North Korea",
    "Democratic Republic of the Congo" : "Congo, Dem. Rep.",
    "Lao People's Democratic Republic" : "Lao",
    "Saint Helena" : "St. Helena",
    "Saint Kitts and Nevis" : "St. Kitts and Nevis",
    "Saint Lucia" : "St. Lucia",
    "Saint Vincent and the Grenadines" : "St. Vincent and the Grenadines",
    "Syrian Arab Republic" : "Syria",
    "Viet Nam" : "Vietnam",
    "West Bank and Gaza Strip" : "Palestine" 
}

# dependent vairable, aid
aid = pd.read_csv('raw_data/TABLE2A_27032021155333179.csv')
# filter for target_year 
aid = aid[aid["Year"] == target_year]

# remove subtotals (recipient name contains 'Total')
for row_index in aid.index:
    if "Total" in aid["Recipient"][row_index]:
        aid = aid.drop(row_index)

# remove unused columns
for col_name in aid.columns:
    if col_name != "Recipient" and col_name != "Value":
        del aid[col_name]
# clean up country names using country_mapping dictionary
aid["Recipient"].replace(country_mapping, inplace=True)
# add a column for % of total, then delete the original Value column
aid["aid_percentage"] = aid["Value"] / aid["Value"].sum() * 100
del aid["Value"]
# remove rows where country is not specified
# note: we do this after the percentage calculation
for row_index in aid.index:
    if "region" in aid["Recipient"][row_index]:
        aid = aid.drop(row_index)
    elif "unspecified" in aid["Recipient"][row_index]:
        aid = aid.drop(row_index)
# set the recipient country column as the index
aid.set_index(keys="Recipient", inplace=True)

# read security excel file, filter for relevant target_year and columns
security = pd.read_excel('raw_data/SFIv2018.xls')

# filter for target_year
security = security[security["year"] == target_year]

# remove unused columns
for col_name in security.columns:
    if col_name != "country" and col_name != "seceff":
        del security[col_name]
# clean up country names using country_mapping dictionary
security["country"].replace(country_mapping, inplace=True)
# set the country column as the index
security.set_index(keys="country", inplace=True)

# create a master dataframe: merged_frame
# inner join all data except aid and commonwealth
merged_frame = security

def csv_to_df(filename, target_year):
    """takes a csv file and returns a dataframe with index = countries and columns = target years"""
    df = pd.read_csv("raw_data/" + filename + ".csv", index_col=0) # index is column 0 country, column names are years (first row)
    # delete all columns but the target year
    for col_name in df.columns:
        if col_name != str(target_year):
            del df[col_name]
    return df

# a dictionary with all csv files and the column names
feature_csvfiles = {
    "life_exp": "life_expectancy_years",
    "calories": "food_supply_kilocalories_per_person_and_day",
    "gdp_pc": "gdppercapita_us_inflation_adjusted",
    "pol_rights": "polrights_fh",
    "civil_lib": "cliberities_fh",
    "mil_exp": "military_expenditure_percent_of_gdp",
    "imports": "imports_percent_of_gdp",
    "popn": "population_total"
    }

def merge_outer(df0, df1) :
    return pd.merge(df0, df1, left_index = True, right_index = True, how = "outer")

# inner join with everything else. for each file in the dictionary, add to the merged_frame
for feature_name, filename in feature_csvfiles.items():
    df = csv_to_df(filename, target_year) # create the DF for target_year
    df.columns = [feature_name] # rename the column
    merged_frame = merge_outer(merged_frame, df) # add to merged_frame

def read_excel_country_list(feature_name, filename):
    """takes an excel file with a single column list of countries and returns a DF with countries and 1's"""
    df = pd.read_excel('raw_data/'+ filename + '.xlsx', names = ["country"], header=None)
    df[feature_name] = 1
    df.set_index(keys="country", inplace=True)
    return df

commonwealth = read_excel_country_list("commonwealth", "commonwealth_members")

# outer join merged_frame with aid and commonwealth. Any NaNs will be filled with zeros
merged_frame = merge_outer(commonwealth, merged_frame)
merged_frame = merge_outer(aid, merged_frame)

total_nulls = merged_frame.isnull().sum()
total_nulls.to_csv("null_values.csv")
# set NaN values to zero for aid and commonwealth
# the country is either in the commonwealth (1) or not (0)
merged_frame["commonwealth"].replace({np.nan: 0}, inplace=True)
# the country either receives aid (>0) or not (0)
merged_frame["aid_percentage"].replace({np.nan: 0}, inplace=True)

# set NaN values to the mean for the other features
for feature in ["seceff", "life_exp", "calories", "gdp_pc", "pol_rights", "civil_lib", "mil_exp", "imports", "popn"]:
    merged_frame[feature] = merged_frame[feature].fillna(merged_frame[feature].mean())

# *********** log transformation ***********
# add log transformation column if it improves the skew
# return True if a column was added, otherwise false
def add_log_if_improves_skew(feature, df) :
    """Compares the skew for the feature with its log transformation. Adds a log column and returns True if it improves the skew"""
    featureData = df[feature]  
    logged = np.log(featureData)
    if abs(logged.skew()) >= abs(featureData.skew()) :
        return False
    df[feature+"_log"] = logged
    return True

# log transformation of continuous variables
# skip mil_exp due to errors (cannot log zero)
to_drop = []
for feature in ["life_exp", "calories", "gdp_pc", "imports", "popn"]:
    column_added = add_log_if_improves_skew(feature, merged_frame)
    if column_added :
        to_drop.append(feature)

# save as csv
merged_frame.to_csv('merged_frame.csv')

# drop features which I want to log
clean_df = merged_frame.drop(to_drop, axis=1)

# *********** outliers ***********
# identify and remove outliers
# the rows where any of the values are > mean + 3 * the s.d. for the column
for feature in clean_df.columns[1:]:
    mean = clean_df[feature].mean()
    stdx3 = 3 * clean_df[feature].std()
    min = mean - stdx3
    max = mean + stdx3
    for row_index in clean_df.index:
        if clean_df[feature][row_index] > max:
            clean_df = clean_df.drop(row_index)
        elif clean_df[feature][row_index] < min:
            clean_df = clean_df.drop(row_index)

# *********** normalise ***********
# define x and y
features_df = clean_df.drop("aid_percentage", axis=1)
targets_df = clean_df["aid_percentage"]

# scale to range 0-1
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(features_df,targets_df)
scaled_features_df = pd.DataFrame(scaled_array, columns = features_df.columns, index = features_df.index)

# save as csv
targets_df.to_csv('targets_df.csv')
scaled_features_df.to_csv('scaled_features_df.csv')

