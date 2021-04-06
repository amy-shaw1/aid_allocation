import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# before data transformation (includes original and log columns)
merged_frame = pd.read_csv('merged_frame.csv', index_col=0)
# after data transformation
scaled_features_df = pd.read_csv('scaled_features_df.csv', index_col=0)
targets_df = pd.read_csv('targets_df.csv', index_col=0)
# combine targets and scaled features
scaled_df = pd.merge(targets_df, scaled_features_df, left_index = True, right_index = True)


# *** Plot histograms to show the original and the log transformation for the 3 features ***
feature_names_log = ["popn", "popn_log", "imports", "imports_log", "gdp_pc", "gdp_pc_log"]

fig1, ax1 = plt.subplots(3, 2)
count = 0
for ax in ax1.flatten() :
    ax.hist(merged_frame[feature_names_log[count]])
    ax.set_title(feature_names_log[count])
    count += 1
    if count >= len(feature_names_log):
        break
plt.tight_layout()
plt.savefig('fig1_histograms.png')

# *** Plot histogram to show distribution of target variable aid_percentage ***
# filtered for aid_percentage > 0
aid_recipients_df = scaled_df
for row_index in aid_recipients_df.index:
    if aid_recipients_df["aid_percentage"][row_index] <=0:
        aid_recipients_df = aid_recipients_df.drop(row_index)

fig2 = plt.figure(figsize=(5,5))
plt.hist(aid_recipients_df["aid_percentage"], bins=20)
plt.xlabel("aid_percentage")
plt.ylabel("frequency")
plt.savefig('fig2_aid_hist.png')

# Create a dictionary with keys = aid_percentage and values = True/False
aid_bool_dict = {}
for row_index in targets_df.index:
    aid_percent = targets_df["aid_percentage"][row_index] 
    if aid_percent <= 0.0:
        aid_bool_dict[aid_percent] = False
    else:
        aid_bool_dict[aid_percent] = True
# Apply the map() function to the aid_percentage column
targets_df["aid_bool"] = targets_df["aid_percentage"].map(aid_bool_dict)
# group by the new bool column
aid_bool_counts = targets_df.groupby("aid_bool").size()

# *** Plot pie chart to show % of countries who are recipients of aid ***
fig3 = plt.figure(figsize=(5,5))
plt.pie(aid_bool_counts,labels=('No aid received','Aid received'), autopct='%1.1f%%')
#plt.suptitle("Proportion of countries receiving aid from the UK")
plt.savefig('fig3_aid_bool.png')


# *** Scatter plots to show the relationship between each feature and the target
fig4a_features = ["commonwealth", "seceff", "pol_rights","life_exp"]
fig4b_features = [ "calories",  "gdp_pc_log", "popn_log","imports_log"]

fig4a, ax4a = plt.subplots(2, 2)
count = 0
for ax in ax4a.flatten():
    feature = fig4a_features[count]
    ax.scatter(scaled_df[feature],scaled_df["aid_percentage"])
    ax.set_title(feature)
    count += 1
plt.tight_layout()
plt.savefig('fig4a_scatter.png')

fig4b, ax4b = plt.subplots(2, 2)
count = 0
for ax in ax4b.flatten():
    feature = fig4b_features[count]
    ax.scatter(scaled_df[feature],scaled_df["aid_percentage"])
    ax.set_title(feature)
    count += 1
plt.tight_layout()
plt.savefig('fig4b_scatter.png')