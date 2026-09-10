# %%
import pandas as pd
import numpy as np

# %%
df_tranpl = pd.read_csv("../data/aligned/aligned_2year_transpl.csv")
df_dialysis = pd.read_csv("../data/aligned/aligned_2year_dialysis.csv")
df_egfr = pd.read_csv("../data/aligned/aligned_2year_gfr_est_bool.csv")
df_egfr_wilson = pd.read_csv("../data/aligned/aligned_2year_gfr_est_w_bool.csv")
df_death = pd.read_csv("../data/aligned/aligned_2year_combineddeath.csv")
df_age = pd.read_csv("../data/aligned/aligned_2year_age.csv")
df_score = pd.read_csv("../data/aligned/aligned_2year_kidneyscore_bool.csv")
df_egfr_values = pd.read_csv("../data/aligned/aligned_2year_gfr_est.csv")
df_score_values = pd.read_csv("../data/aligned/aligned_2year_kidneyscore.csv")

# %% [markdown]
# #### Outcome creation

# %%
# frames = [df_tranpl, df_dialysis, df_egfr]
frames = [df_tranpl, df_dialysis, df_egfr_wilson]
# frames = [df_death]
# frames = [df_tranpl, df_dialysis, df_egfr_wilson, df_death[df_death.columns[:-1]]]



outcomes = None
for frame in frames:

    cols = [col for col in frame.columns if col != "ID"]
    
    vals = frame[cols].values.astype(float)

    if outcomes is None:
        outcomes = np.zeros_like(vals)
    outcomes += vals




# %%
col_out = [col.replace("GFR_est", "Outcome") for col in cols]
df_outcome = (pd.DataFrame(data=outcomes,columns=col_out) > 0)*1

# %%
df_outcome.head()

# %% [markdown]
# ##### Number of individuals with ESKD that died

# %%
cols_death, cols_out = [col for col in df_death.columns if col!="ID"], [col for col in df_outcome.columns if col!="ID"]
cond_inter_death_out = (df_death[cols_death].sum(axis=1) > 0)  & (df_outcome[cols_out].sum(axis=1) > 0) 
cond_inter_death_out.sum()

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# #### ESKD plots

# %%
plt.figure(figsize=(6, 5))
df_outcome[col_out[:-1]].sum().plot(figsize=(12,5),)
plt.xlabel("Month of Outcome")
plt.ylabel("Number of ESKD events")
# plt.show()

plt.savefig('images/outcomes_over_time.pdf', dpi=600, bbox_inches='tight')
# Show the plot
plt.close()

# %% [markdown]
# # Score - predictions

# %%
col_score = [col for col in df_score.columns if col!= "ID"]
df_score[col_score].sum()

# %%
df_score[col_score[:-1]].sum().plot(figsize=(12,5),)
plt.xlabel("Month of Prediction")
plt.xticks(rotation=20, ha='right')
plt.ylabel("Number of Predicted ESKD events")
# plt.show()

plt.savefig('images/preds_over_time.pdf', dpi=600, bbox_inches='tight')
# Show the plot
plt.close()

# %% [markdown]
# # finding first occurence of outcome or prediction

# %%
def reset_after_ones(row, window=24):
    row = row.to_numpy().copy()  # work on array copy for speed
    i = 0
    while i < len(row):
        if row[i] == 1:
            # zero out the next 'window' cells
            row[i+1:i+1+window] = 0
            i += window + 1  # skip over the window
        else:
            i += 1
    return pd.Series(row)

# Example usage:
# df = pd.DataFrame(np.random.randint(0, 2, (5, 50)))
result_preds = df_score[col_score].apply(reset_after_ones, axis=1)
result_preds.columns = col_score

# %%
result_preds[col_score[:-1]].sum().plot(figsize=(12,5),)
plt.xlabel("Month of Prediction")
plt.xticks(rotation=20, ha='right')
plt.ylabel("Number of Predicted ESKD events")
# plt.show()

plt.savefig('images/preds_only_first_over_time.pdf', dpi=600, bbox_inches='tight')
# Show the plot
plt.close()

# %%
result_out = df_outcome[col_out].apply(reset_after_ones, axis=1)
result_out.columns = col_out

# %%
result_out[col_out].sum().plot(figsize=(12,5),)
plt.xlabel("Month of Outcome")
plt.ylabel("Number of ESKD events")
# plt.show()

plt.savefig('images/outcomes_only_first_over_time.pdf', dpi=600, bbox_inches='tight')

plt.close()

# %% [markdown]
# # Performance measure

# %%
import numpy as np
import pandas as pd

def enforce_cooldown(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """After each 1 in a row, zero out the next `window` cells."""
    arr = df.astype(bool).to_numpy()
    for r in range(arr.shape[0]):
        i = 0
        row = arr[r]
        while i < row.size:
            if row[i]:
                row[i+1:i+1+window] = False
                i += window + 1
            else:
                i += 1
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def first_one_idx_bool_array(a: np.ndarray) -> np.ndarray:
    """Return first-True index per row; np.inf if no True."""
    first = a.argmax(axis=1).astype(float)
    none_mask = ~a.any(axis=1)
    first[none_mask] = np.inf
    return first

def classify_with_window(pred_df: pd.DataFrame,
                         outcome_df: pd.DataFrame,
                         window: int = 24) -> pd.DataFrame:
    # align columns and apply cooldown to BOTH
    pred_cool   = enforce_cooldown(pred_df, window)
    out_cool    = enforce_cooldown(outcome_df, window)

    # compute first indices
    pred_first  = first_one_idx_bool_array(pred_cool.to_numpy())
    out_first   = first_one_idx_bool_array(out_cool.to_numpy())

    lead = pred_first - out_first  # negative => predicted earlier

    # labels (add a 'tie' class if they hit the same column)
    conds = [
        np.isfinite(pred_first) & np.isfinite(out_first) & (pred_first <  out_first),
        np.isfinite(pred_first) & np.isfinite(out_first) & (pred_first >  out_first),
        np.isfinite(pred_first) & np.isfinite(out_first) & (pred_first == out_first),
        np.isinf(pred_first)    & np.isfinite(out_first),
        np.isfinite(pred_first) & np.isinf(out_first),
        np.isinf(pred_first)    & np.isinf(out_first),
    ]
    choices = ["earlier", "later", "tie", "never_predicted", "false_positive", "no_event"]
    label = np.select(conds, choices, default="unknown")

    res = pd.DataFrame({
        "pred_first_idx": pred_first,
        "outcome_first_idx": out_first,
        "lead_pred_minus_outcome": lead,
        "label": label
    }, index=outcome_df.index)

    return res

# --- Usage ---
summary = classify_with_window(result_preds[col_score[:]], result_out[col_out[:]], window=24)
summary["label"].value_counts()


# %%
summary["label"].value_counts().sum()

# %%
tp = summary["label"].value_counts()["earlier"] + summary["label"].value_counts()["tie"]
tn = summary["label"].value_counts()["no_event"]
fp = summary["label"].value_counts()["false_positive"]
fn = summary["label"].value_counts()["never_predicted"] + summary["label"].value_counts()["later"]

# %%
print(" TP: ",tp, "\n TN: ", tn, "\n FN: ", fn, "\n FP: ", fp)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Construct the confusion matrix
conf_matrix = np.array([[tn, fp], [fn, tp]])

# Create a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])

# Add titles and labels
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')

plt.savefig('images/CM.pdf', dpi=600, bbox_inches='tight')
# Show the plot
plt.close()


# %%
rec, prec, acc, spec = tp/(tp+fn), tp/(fp+tp), (tn+tp)/(tp+fp+tn+fn), tn/(fp+tn)
f1 = 2*prec*rec/(prec+rec)
print("Recall: ", rec,"\n Precision: ", prec, "\n Accuracy: ", acc, "\n F1-Score: ", f1 , "\n Specificty: ", spec)

# %%
# plt.figure(figsize=(6, 5))
summary.loc[np.isfinite(summary["pred_first_idx"]) & np.isfinite(summary["outcome_first_idx"]),"pred_first_idx"].hist(bins=25)
plt.title("Time of model prediction")
plt.xlabel("Months/in integers")
plt.ylabel("Frequency/counts")
plt.xlim(0,28)
# plt.show()

plt.savefig('images/time_of_preds.pdf', dpi=600, bbox_inches='tight')

plt.close()


# %%
summary.loc[np.isfinite(summary["pred_first_idx"]) & np.isfinite(summary["outcome_first_idx"]),"pred_first_idx"].describe()

# %%
# plt.figure(figsize=(6, 5))
summary.loc[np.isfinite(summary["pred_first_idx"]) & np.isfinite(summary["outcome_first_idx"]),"pred_first_idx"].hist(bins=25, cumulative=True, density=True)
plt.title("Cummulative density function \n of model's time of prediction")
plt.xlabel("Months/in integers")
plt.ylabel("Percentage - %")
plt.xlim(0,28)
# plt.show()

plt.savefig('images/cum_time_of_preds.pdf', dpi=600, bbox_inches='tight')

plt.close()


# %%
series = summary.loc[np.isfinite(summary["pred_first_idx"]) & np.isfinite(summary["outcome_first_idx"]),"pred_first_idx"]
pct_at_5_pred = (series <= 5).mean() * 100
print(f"Percentage ≤ 5: {pct_at_5_pred:.2f}%")

# %%
# plt.figure(figsize=(6, 5))
summary.loc[np.isfinite(summary["outcome_first_idx"]),"outcome_first_idx"].hist(bins=24)
plt.title("Time of ESKD outcome")
# plt.title("Time of Death outcome")
plt.xlabel("Months/in integers")
plt.ylabel("Frequency/counts")
plt.xlim(0,28)
# plt.show()

plt.savefig('images/time_of_out.pdf', dpi=600, bbox_inches='tight')

plt.close()

# %%
summary.loc[np.isfinite(summary["outcome_first_idx"]),"outcome_first_idx"].describe()

# %%
summary.loc[np.isfinite(summary["outcome_first_idx"]),"outcome_first_idx"].hist(bins=26, cumulative=True, density=True)
plt.title("Cummulative density function of Time of ESKD outcome")
# plt.title("Time of Death outcome")
plt.xlabel("Months/in integers")
plt.ylabel("Frequency/counts")
plt.xlim(0,28)
# plt.show()

plt.savefig('images/cum_time_of_out.pdf', dpi=600, bbox_inches='tight')

plt.close()

# %%
series = summary.loc[np.isfinite(summary["outcome_first_idx"]),"outcome_first_idx"]
pct_at_5_out = (series <= 5).mean() * 100
print(f"Percentage ≤ 5: {pct_at_5_out:.2f}%")

# %%
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Your values
# ----------------------------
labels = ["Model Prediction Time", "ESKD Outcome Time"]
means = [summary.loc[np.isfinite(summary["pred_first_idx"]) & np.isfinite(summary["outcome_first_idx"]),"pred_first_idx"].describe()['mean'], summary.loc[np.isfinite(summary["outcome_first_idx"]),"outcome_first_idx"].describe()['mean']]
stds  = [summary.loc[np.isfinite(summary["pred_first_idx"]) & np.isfinite(summary["outcome_first_idx"]),"pred_first_idx"].describe()['std'], summary.loc[np.isfinite(summary["outcome_first_idx"]),"outcome_first_idx"].describe()['std']]

x = np.arange(len(labels))

# ----------------------------
# Plot: Mean ± Standard Deviation
# ----------------------------
bars = plt.bar(
    x, means, yerr=stds, capsize=8,
    label="Mean ± SD"
)

plt.xticks(x, labels, fontsize=12)
plt.ylabel("Time / Months", fontsize=12)
plt.title("Mean Time with Standard Deviation", fontsize=14)

# ----------------------------
# Add value labels on bars
# ----------------------------
for bar, mean, std in zip(bars, means, stds):
    mean_sf = f"{mean:.2g}"
    std_sf  = f"{std:.2g}"
    
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{mean_sf} ± {std_sf}",
        ha="left",
        va="bottom", 
        fontsize=18
    )

# ----------------------------
# Legend & Grid
# ----------------------------
plt.legend()
plt.grid(True)

# plt.show()

plt.savefig('images/bar_times_comp.pdf', dpi=600, bbox_inches='tight')

plt.close()


# %%
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Replace with YOUR percentages
# ----------------------------
pct_model   = pct_at_5_pred   # example: % ≤ t for model
pct_outcome = pct_at_5_out   # example: % ≤ t for outcome

labels = ["Model: % of cases predicted", "Outcome: % of cases occured"]
values = [pct_model, pct_outcome]

x = np.arange(len(labels))

# ----------------------------
# Plot
# ----------------------------
plt.figure()

bars = plt.bar(x, values)

plt.xticks(x, labels, fontsize=12)
plt.ylabel("Percentage (%)",  fontsize=12)
plt.title("Percentage of cases by the first 5 Months", fontsize=16)

# ----------------------------
# Add value labels on bars
# ----------------------------
for bar, val in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{val:.2f}%",
        ha="center",
        va="bottom", 
        fontsize=18
    )

plt.ylim(0, 100)
plt.grid(True)
# plt.show()

plt.savefig('images/bar_first_5_months.pdf', dpi=600, bbox_inches='tight')

plt.close()

# %%
# tie cases
result_preds.loc[result_out[col_out[0]] > 0,result_preds.columns[0]].sum()

# %%
result_preds[result_preds[col_score[:-1]].sum(axis=1) > 0].shape

# %%
result_preds[result_preds[col_score[:-1]].sum(axis=1) > 0].index.isin(result_out[result_out[col_out].sum(axis=1) > 0].index).sum()

# %% [markdown]
# # Bias and Fairness Analysis

# %% [markdown]
# #### Create df fairness and other equiboots inputs
# 
# y_pred: ["false_positive", "earlier", "later", "tie"]
# 
# y_true: ["earlier", "later", "tie", "never_predicted"]

# %% [markdown]
# #### Getting prediction closest to the outcome but before the outcome if missing picking 1st prediction

# %%
# 1. Map columns to absolute numerical position indices
# Assuming both dataframes share the exact same chronological time columns
df2 = result_out
df1 = result_preds
col_positions = np.arange(len(df2.columns))

# 2. Vectorized: Find the column index of the FIRST '1' in df2 per row
# If a row has no 1, idxmax returns 0, so we create a mask to protect data integrity
df2_has_one = (df2 == 1).any(axis=1)
df2_first_pos = (df2 == 1).to_numpy().argmax(axis=1)

# 3. Vectorized: Extract all coordinate locations where df1 equals 1
# This yields row and column index arrays matching your active entries
df1_row_idx, df1_col_idx = np.where(df1 == 1)

# 4. Compute the closest match using a grouped matrix reduction
# We initialize an array to hold the closest column index found in df1
closest_df1_col_indices = np.full(len(df2), np.nan)

# Process via a fast grouped loop over active rows to find global minimums
for row_num in np.unique(df1_row_idx):
    if not df2_has_one.iloc[row_num]:
        continue  # Skip if master anchor doesn't exist
        
    # Get the target column position anchor from df2
    target_anchor = df2_first_pos[row_num]
    
    # Isolate all column index positions where df1 has a '1' for THIS specific row
    available_df1_positions = df1_col_idx[df1_row_idx == row_num]
    
    if len(available_df1_positions) > 0:
        # 1. Calculate raw directional distances (df1_pos - target_anchor)
        # Negative = df1 happened BEFORE df2 anchor
        # Zero     = df1 happened AT THE SAME TIME as df2 anchor
        # Positive = df1 happened AFTER df2 anchor (we want to discard these)
        raw_distances = available_df1_positions - target_anchor
        
        # 2. Filter for indices that are smaller than or equal to the anchor
        valid_historical_mask = raw_distances <= 0
        valid_positions = available_df1_positions[valid_historical_mask]
        valid_distances = raw_distances[valid_historical_mask]
        
        if len(valid_positions) > 0:
            # 3. Find the one closest to the anchor 
            # Since these are negative/zero numbers, the closest to 0 is the MAX value
            closest_match_pos = valid_positions[np.argmax(valid_distances)]
            
            closest_df1_col_indices[row_num] = closest_match_pos
        else:
            # If df1 only has occurrences AFTER df2, mark it as 0 (per your preference)
            closest_df1_col_indices[row_num] = 0

# Wrap it into a clean, searchable tracking Series
df_closest_mapping = pd.Series(closest_df1_col_indices, index=df2.index)
df_closest_mapping.fillna(0, inplace=True)


# %%
# 1. Extract the raw NumPy array from your DataFrame and scale it once
# Doing the division / 100 on the whole matrix at once is incredibly fast in C
matrix_scaled = df_score_values.loc[:,df_score_values.columns[1:]].to_numpy() / 100

# 2. Get your row indices (0, 1, 2, ..., 48000)
row_indices = np.arange(len(df_score_values))

# 3. Convert your mapped column indices to a clean, flat integer NumPy array
col_indices = df_closest_mapping.astype(int).to_numpy()

# 4. Use advanced NumPy indexing to grab exactly one specific column value per row
# matrix_scaled[row, col] extracts a 1D array of values instantly
y_prob = matrix_scaled[row_indices, col_indices]

# %%
y_pred = (summary.label.isin(["false_positive", "earlier", "later", "tie"])) * 1
y_true = (summary.label.isin(["earlier", "later", "tie", "never_predicted"])) * 1

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import numpy as np

# Assuming y_true and y_prob are already defined
# y_true: binary labels (0 or 1)
# y_prob: probabilities for class 1

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# 2. PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)

plt.subplot(1, 3, 2)
plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
plt.title('PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# 3. Calibration Curve
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.subplot(1, 3, 3)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.title('Calibration Curve')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()

plt.tight_layout()
# plt.show()

plt.savefig('images/perf_cruves.pdf', dpi=600, bbox_inches='tight')



# %%
df_fair_raw = pd.read_csv("../data/CKD_Fairness_vars.csv")
df_fair_raw.head()

# %%
df_fair_filt = pd.merge(df_age["ID"],df_fair_raw, how="left", on="ID")
df_fair_filt.head()

# %%
df_fair_filt.shape

# %% [markdown]
# # Adding demographics a few months before or after we start predicting

# %%
df_fair_filt["age"] = df_age['Age_Month_1']
df_fair_filt.head()

# %%
# load prior egfrs
# df_egfr_values_prior = pd.read_csv("../data/processed/egfr_lj_scores_prior_model.csv")
df_egfr_values_prior = pd.read_csv("../data/CKD_GFR_per_pat_pivot_month_min.csv")
df_shift_positions = pd.read_csv("../data/aligned/shift_positions.csv")

df_egfr_values_prior_filt = pd.merge(df_fair_filt["ID"],df_egfr_values_prior, how="left", on="ID")

# %%
df_egfr_values_prior_filt.head(3)

# %%
egfrs_prior_test = []
egfrs_period_of_testing = df_egfr_values_prior_filt.iloc[:,6:18] # 6 is July 2023 and 18 is inclding May 2024

# faster numpy implementation in next cell
for index, row in df_shift_positions.iterrows():
    row_indx, col_indx = row["row_indx"], row["col_index"]
    egfrs_prior_test.append(egfrs_period_of_testing.iloc[row_indx, : col_indx+1].mean())

df_fair_filt["egfr"]  = egfrs_prior_test

# special case probably misalignment of test perhaps on last date of the month and shifted to next month
# lets use average of period, otherwise they will be missing values

df_fair_filt.loc[df_fair_filt["egfr"].isnull(), "egfr"] = df_egfr_values_prior_filt.iloc[:,1:18].mean(axis=1)

# %%
# df_fair_filt["egfr"] = egfrs_prior_test
df_fair_filt.head(3)

# %%
race_cats = df_fair_filt.FirstRace.value_counts().index

white = [i for i in race_cats if 'White' in i 
         or "European" in i
         or "Armenian" in i]
asian = [i for i in race_cats if 'Asian' in i or "Chinese" in i 
         or "Filipino" in i
         or "Japanese" in i
         or "Korean" in i
         or "Indian (India)" in i
         or "Taiwanese" in i
         or "Vietnamese" in i
         or "Thai" in i
         or "Sri Lankan" in i
         or "Indonesian" in i
         ]
black = [i for i in race_cats if 'Black' in i or "African" in i]
native = [i for i in race_cats if 'Alaska Native' in i 
          or "native" in i
          or "Samoan" in i]
islanders = [i for i in race_cats if 'Islander' in i]
other = [i for i in race_cats if i not in white and 
                                 i not in asian and 
                                 i not in black and
                                 i not in native and
                                 i not in islanders]

df_fair_filt.loc[df_fair_filt.FirstRace.isin(white),'Race'] = 'white'
df_fair_filt.loc[df_fair_filt.FirstRace.isin(asian),'Race'] = 'asian'
df_fair_filt.loc[df_fair_filt.FirstRace.isin(black),'Race'] = 'black'
df_fair_filt.loc[df_fair_filt.FirstRace.isin(native),'Race'] = 'native'
df_fair_filt.loc[df_fair_filt.FirstRace.isin(islanders),'Race'] = 'islanders'
df_fair_filt.loc[df_fair_filt.FirstRace.isin(other),'Race'] = 'other_race'

# %%
df_fair_filt.head()

# %%
# 1. Custom bins for Age (Adults Only)
age_bins = [18, 40, 65, 80, 101]  # Updated to exclude children, and now just adult categories
age_labels = ['Young Adult', 'Middle-Aged', 'Older Adult', 'Senior']

df_fair_filt['age_group'] = pd.cut(df_fair_filt['age'], bins=age_bins, labels=age_labels, right=False).astype(object)

# 2. Custom bins for SVI (Social Vulnerability Index)
svi_bins = [0, 33, 66, 100]  # Vulnerability bins
svi_labels = ['Low Vulnerability', 'Moderate Vulnerability', 'High Vulnerability']

df_fair_filt['SVI_category'] = pd.cut(df_fair_filt['SVI_Rank'], bins=svi_bins, labels=svi_labels, right=True).astype(object)
# Add 'Missing' to categories, then fill NaNs
df_fair_filt['SVI_category'] = df_fair_filt['SVI_category'].fillna('Missing')

# 3. Custom bins for eGFR (CKD Stages)
egfr_bins = [0, 15, 29, 59, 89, 300]  # eGFR bins (CKD Staging)
egfr_labels = ['Stage 5 (Kidney Failure)', 'Stage 4 (Severely Decreased)', 
               'Stage 3 (Moderately Decreased)', 'Stage 2 (Mildly Decreased)', 
               'Stage 1 (Normal or High)']

df_fair_filt['eGFR_stage'] = pd.cut(df_fair_filt['egfr'], bins=egfr_bins, labels=egfr_labels, right=False).astype(object)

# Display the categorized data
df_fair_filt.head()

# %%
df_fair_filt['SVI_category'].value_counts(dropna=False)

# %%
# df_fair_filt.loc[df_fair_filt['SVI_category'] == "NaN","SVI_Rank"]
df_fair_filt['age'].describe()

# %%
df_fair_filt['age_group'].value_counts(dropna=False)

# %%
df_fair_filt['eGFR_stage'].value_counts(dropna=False)

# %%
df_fair_filt.shape

# %%
df_egfr_values_prior.shape

# %%
df_fair_filt.head()

# %%
df_egfr_values_prior_filt[df_fair_filt['eGFR_stage'].isnull()]

# %% [markdown]
# # EDA toolkit to generate table 1 
# 
# - creating a copy of only vars of interest
# - adding outcome

# %%
cat_cols = ["Race", "Ethnicity", "Sex", "age_group", "SVI_category", "eGFR_stage"]
cont_cols = ["SVI_Rank", "age", "egfr"] 
outcome_col = ["ESKD event"]

# %%
df_table1 = df_fair_filt[cat_cols+cont_cols].copy()
df_table1[outcome_col[0]] = y_true

# %%
df_table1.rename(columns={"Race": "Race", "age_group": "Age group", "SVI_category": "SVI group", "eGFR_stage": "eGFR stage", "egfr": "eGFR",
                          "SVI_Rank": "SVI", "age": "Age", }, inplace=True)

# Define your groups
mapping = {
    'Not Hispanic or Latino': 'Non-Hispanic',
    'Hispanic or Latino': 'Hispanic',
    'Mexican, Mexican American, Chicano/a': 'Hispanic',
    'Other Hispanic, Latino/a, or Spanish origin': 'Hispanic',
    'Cuban': 'Hispanic',
    'Puerto Rican': 'Hispanic',
    'Choose Not to Answer': 'Unknown',
    'Unknown': 'Unknown'
}

# Apply the change
df_table1['Ethnicity'] = df_table1['Ethnicity'].map(mapping)
df_fair_filt['Ethnicity'] = df_fair_filt['Ethnicity'].map(mapping)

df_table1['ESKD event'] = df_table1['ESKD event'].replace({0: "Non-ESKD", 1: "ESKD"})

# %%
df_table1["Age group"].value_counts()

# %%
from eda_toolkit import generate_table1

# Generate Table 1 comparing income groups
p_value_table_1_cont, p_value_table_1_cat = generate_table1(
    df=df_table1,
    value_counts=True,
    # include_types="categorical",
    export_markdown=True,
    groupby_col="ESKD event",
    combine=False,
    drop_columns=[
        # "Missing (n)",
        # "Missing (%)",
        "ESKD event",   # drop the raw income column itself
        "Type",     # drop internal type metadata
        "Mode",     # drop mode metadata
    ],
    
    drop_variables="ESKD event",
)

# %%
print(p_value_table_1_cat)

# %% [markdown]
# ### Launching equiboots

# %%
import equiboots as eqb


# Create fairness DataFrame
fairness_df = df_fair_filt[["Race", "Sex", "Ethnicity", "SexualOrientation", "age_group", "SVI_category", "eGFR_stage"]].reset_index()

eq = eqb.EquiBoots(
    y_true=y_true,
    y_prob=y_prob,
    y_pred=y_pred,
    fairness_df=fairness_df,
    fairness_vars=["Race", "Sex", "Ethnicity", "SexualOrientation", "age_group", "SVI_category", "eGFR_stage"],
    group_min_size=2500
)

# grouping by variables' groups (e.g., Male, Female, etc)
eq.grouper(groupings_vars=["Race", "Sex", "Ethnicity", "SexualOrientation", "age_group", "SVI_category", "eGFR_stage"])

# %%
# slicing data by race
sliced_race_data = eq.slicer("Race")
# slicing and generating metrics for sex
sliced_sex_data = eq.slicer("Sex")
sliced_ethn_data = eq.slicer("Ethnicity")
sliced_sex_or_data = eq.slicer("SexualOrientation")
sliced_age_data = eq.slicer("age_group")
sliced_svi_data = eq.slicer("SVI_category")
sliced_egfr_data = eq.slicer("eGFR_stage")

race_metrics = eq.get_metrics(sliced_race_data)
sex_metrics = eq.get_metrics(sliced_sex_data)
ethn_metrics = eq.get_metrics(sliced_ethn_data)
sex_or_metrics = eq.get_metrics(sliced_sex_or_data)
age_metrics = eq.get_metrics(sliced_age_data)
svi_metrics = eq.get_metrics(sliced_svi_data)
egfr_metrics = eq.get_metrics(sliced_egfr_data)

# %%
## generating statistical significnace in tests
test_config = {
    "test_type": "chi_square",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
}

# stat test race
stat_test_results_race = eq.analyze_statistical_significance(
    race_metrics, "Race", test_config
)

# stat test sex
stat_test_results_sex = eq.analyze_statistical_significance(
    sex_metrics, "Sex", test_config
)

stat_test_results_ethn = eq.analyze_statistical_significance(
    ethn_metrics, "Ethnicity", test_config
)

stat_test_results_sex_or = eq.analyze_statistical_significance(
    sex_or_metrics, "SexualOrientation", test_config
)

stat_test_results_age = eq.analyze_statistical_significance(
    age_metrics, "age_group", test_config
)

stat_test_results_svi = eq.analyze_statistical_significance(
    svi_metrics, "SVI_category", test_config
)

stat_test_results_egfr = eq.analyze_statistical_significance(
    egfr_metrics, "eGFR_stage", test_config
)

# %%
overall_stat_results = {
    "Sex": stat_test_results_sex,
    "Race": stat_test_results_race,
    "Ethnicity": stat_test_results_ethn,
    "SexualOrientation": stat_test_results_sex_or,
    "Age group": stat_test_results_age,
    "SVI category": stat_test_results_svi,
    "eGFR stage": stat_test_results_egfr,
}

# %%
plt.rcParams.update({'font.size': 16})
eqb.eq_group_metrics_point_plot(
    group_metrics=[race_metrics, sex_metrics, ethn_metrics, sex_or_metrics, age_metrics, svi_metrics, egfr_metrics],
    metric_cols=[
        "Accuracy",
        "Precision",
        "Recall",
    ],
    category_names=["Race", "Sex", "Ethnicity", "SexualOrientation", "Age group", "SVI category", "eGFR stage"],
    figsize=(30, 15),
    include_legend=True,
    plot_thresholds=(0.9, 1.1),
    raw_metrics=True,
    show_grid=True,
    y_lim=(0, 1),
    statistical_tests=overall_stat_results,
    y_lims={(0, 0): (0, 1.0), (0, 1): (0, 1.0)},
    leg_cols=7,
    save_path='images',
    filename="pe_disp_all_vars_metrics.pdf",
)


# %%
from equiboots.tables import metrics_table

# %%
# ------------------------------------------------------------------------------
# Point Estimate Metrics Tables with Statistical Significance
# ------------------------------------------------------------------------------

# 1. Race
stat_metrics_race_table_point = metrics_table(
    race_metrics, 
    statistical_tests=stat_test_results_race, 
    reference_group="White"
)

# 2. Sex
stat_metrics_sex_table_point = metrics_table(
    sex_metrics, 
    statistical_tests=stat_test_results_sex, 
    reference_group="Male"  # Or your chosen reference demographic, e.g., "Female"
)

# 3. Ethnicity
stat_metrics_ethn_table_point = metrics_table(
    ethn_metrics, 
    statistical_tests=stat_test_results_ethn, 
    reference_group="Not Hispanic or Latino"  # Adjust based on your dataset labels
)

# 4. Sexual Orientation
stat_metrics_sex_or_table_point = metrics_table(
    sex_or_metrics, 
    statistical_tests=stat_test_results_sex_or, 
    reference_group="Straight"  # Adjust based on your dataset labels
)

# 5. Age Group
stat_metrics_age_table_point = metrics_table(
    age_metrics, 
    statistical_tests=stat_test_results_age, 
    reference_group="45-64"  # Or whichever cohort serves as your clinical baseline
)

# 6. Social Vulnerability Index (SVI)
stat_metrics_svi_table_point = metrics_table(
    svi_metrics, 
    statistical_tests=stat_test_results_svi, 
    reference_group="Low Vulnerability"  # Typically the lowest risk quartile/category
)

# 7. Clinical Baseline (eGFR Stage)
stat_metrics_egfr_table_point = metrics_table(
    egfr_metrics, 
    statistical_tests=stat_test_results_egfr, 
    reference_group="Stage 2"  # Baseline reference stage for your outpatient cohort
)

# %% [markdown]
# ### Plots of effect size and Metric for Race

# %%
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_label(text):
    # Remove specific symbols like ▲ and *
    cleaned = re.sub(re.escape('▲'), '', text)
    cleaned = re.sub(re.escape('*'), '', cleaned)
    # Strip any remaining leading/trailing whitespace
    return cleaned.strip()

# Data
groups = [clean_label(key) for key in stat_metrics_race_table_point.columns]
accuracy = stat_metrics_race_table_point.loc[stat_metrics_race_table_point.index.to_series() == "Accuracy",:].iloc[0].tolist()
effect_size = [stat_test_results_race[key]["Accuracy"].effect_size if clean_label(key) in stat_test_results_race.keys() else 0 for key in groups]

groups = [group.replace("_"," ") for group in groups]

x = np.arange(len(groups))
width = 0.35  # width of the bars

# Create grouped bar chart with dual axes to handle different scales properly
fig, ax1 = plt.subplots(figsize=(10, 6.0)) # Optimized size for horizontal legends

color1 = '#1f77b4' # Muted blue
rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.15)

# --- CRITICAL FIX: Removed x-axis label as requested ---
ax1.set_xlabel('', labelpad=0) 

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = '#ff7f0e' # Muted orange
rects2 = ax2.bar(x + width/2, effect_size, width, label='Effect Size', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Effect Size', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# --- CRITICAL FIX: Right axis limits set from 0 to 1 ---
ax2.set_ylim(0, 1.15) 

# --- CRITICAL FIX: Turn grid on ---
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars stay in front of grid lines
ax1.set_axisbelow(True)

# Add values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=15, ha='right', fontsize=11)

# --- CRITICAL FIX: Main Metrics Legend set to horizontal line (ncol=2) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
main_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, frameon=True, shadow=False)
ax1.add_artist(main_legend) # Freeze this legend layer to allow extra ones

# Configure the figure title
plt.title('Race Group-Specific Accuracy and Bias Effect Size (Side-by-Side)', fontsize=13, fontweight='bold', pad=15)

# --- CRITICAL FIX: Stacked horizontal banners placed one below the other using fixed vertical coordinates ---
horizontal_legend_text_sign = r"$\bf{Significance\ Indicators:}$ $*$ Omnibus significance, $\blacktriangle$ Pairwise vs. ref. group"
horizontal_legend_text_eff_size = r"$\bf{Effect\ Size\ Thresholds:}$ Small $\leq$ 0.2, Medium $\leq$ 0.6, Large $>$ 0.6"

props_footer = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)

# Placed horizontally centered, stacked vertically near the bottom canvas window
# fig.text(0.544, -0.05, horizontal_legend_text_sign, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')
fig.text(0.5, 0.00, horizontal_legend_text_eff_size, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')

# Adjust layout to cleanly incorporate the bottom text legend without cutoff
plt.tight_layout()
plt.subplots_adjust(bottom=0.16) 

# plt.savefig('grouped_bar_chart_side_by_side.png', dpi=300)
# plt.close()

print("Grouped bar chart generated successfully.")

plt.savefig('images/bar_chart_accuracy_race_effect.pdf', dpi=600, bbox_inches='tight')


# %%
# PR curves
eqb.eq_plot_group_curves(
    sliced_race_data,
    curve_type="pr",
    subplots=False,
    figsize=(10,8),
    title="Precision-Recall by Race Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="pr_curves_race.pdf"
)

# %%
# ROC curves
eqb.eq_plot_group_curves(
    sliced_race_data,
    curve_type="roc",
    title="ROC AUC by Race Group",
    figsize=(10,8),
    decimal_places=2,
    subplots=False,
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="roc_curves_race.pdf"
    
)

# %%
# calibration curves
eqb.eq_plot_group_curves(
    sliced_race_data,
    curve_type="calibration",
    shade_area=True,
    title="Calibration by Race Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    subplots=False,
    figsize=(10,8),
    save_path="images",
    filename="cal_curves_race.pdf"
)

# %% [markdown]
# #### EGFR

# %%
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_label(text):
    # Remove specific symbols like ▲ and *
    cleaned = re.sub(re.escape('▲'), '', text)
    cleaned = re.sub(re.escape('*'), '', cleaned)
    # Strip any remaining leading/trailing whitespace
    return cleaned.strip()


# Data
groups = [clean_label(key) for key in stat_metrics_egfr_table_point.columns]
accuracy = stat_metrics_egfr_table_point.loc[stat_metrics_egfr_table_point.index.to_series() == "Accuracy",:].iloc[0].tolist()
effect_size = [stat_test_results_egfr[key]["Accuracy"].effect_size if clean_label(key) in stat_test_results_egfr.keys() else 0 for key in groups]

groups = [group.replace("_"," ") for group in groups]

x = np.arange(len(groups))
width = 0.35  # width of the bars

# Create grouped bar chart with dual axes to handle different scales properly
fig, ax1 = plt.subplots(figsize=(10, 6.0)) # Optimized size for horizontal legends

color1 = '#1f77b4' # Muted blue
rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.2)

# --- CRITICAL FIX: Removed x-axis label as requested ---
ax1.set_xlabel('', labelpad=0) 

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = '#ff7f0e' # Muted orange
rects2 = ax2.bar(x + width/2, effect_size, width, label='Effect Size', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Effect Size', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# --- CRITICAL FIX: Right axis limits set from 0 to 1 ---
ax2.set_ylim(0, 1.2) 

# --- CRITICAL FIX: Turn grid on ---
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars stay in front of grid lines
ax1.set_axisbelow(True)

# Add values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=15, ha='right', fontsize=11)

# --- CRITICAL FIX: Main Metrics Legend set to horizontal line (ncol=2) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
main_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, frameon=True, shadow=False)
ax1.add_artist(main_legend) # Freeze this legend layer to allow extra ones

# Configure the figure title
plt.title('eGFR Group-Specific Accuracy and Bias Effect Size (Side-by-Side)', fontsize=13, fontweight='bold', pad=15)

# --- CRITICAL FIX: Stacked horizontal banners placed one below the other using fixed vertical coordinates ---
horizontal_legend_text_sign = r"$\bf{Significance\ Indicators:}$ $*$ Omnibus significance, $\blacktriangle$ Pairwise vs. ref. group"
horizontal_legend_text_eff_size = r"$\bf{Effect\ Size\ Thresholds:}$ Small $\leq$ 0.2, Medium $\leq$ 0.6, Large $>$ 0.6"

props_footer = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)

# Placed horizontally centered, stacked vertically near the bottom canvas window
# fig.text(0.544, -0.15, horizontal_legend_text_sign, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')
fig.text(0.5, -0.10, horizontal_legend_text_eff_size, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')

# Adjust layout to cleanly incorporate the bottom text legend without cutoff
plt.tight_layout()
plt.subplots_adjust(bottom=0.16) 

# plt.savefig('grouped_bar_chart_side_by_side.png', dpi=300)
# plt.close()

print("Grouped bar chart generated successfully.")

plt.savefig('images/bar_chart_accuracy_egfr_effect.pdf', dpi=600, bbox_inches='tight')

# %%
# PR curves
eqb.eq_plot_group_curves(
    sliced_egfr_data,
    curve_type="pr",
    subplots=False,
    figsize=(10,8),
    title="Precision-Recall by eGFR Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="pr_curves_egfr.pdf"
)

# %%
# ROC curves
eqb.eq_plot_group_curves(
    sliced_egfr_data,
    curve_type="roc",
    title="ROC AUC by eGFR Group",
    figsize=(10, 8),
    decimal_places=2,
    subplots=False,
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="roc_curves_egfr.pdf"
)

# %%
# calibration curves
eqb.eq_plot_group_curves(
    sliced_egfr_data,
    curve_type="calibration",
    shade_area=True,
    figsize=(10, 8),
    title="Calibration by eGFR Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    subplots=False,
    save_path="images",
    filename="cal_curves_egfr.pdf"
)

# %% [markdown]
# #### Sex

# %%
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_label(text):
    # Remove specific symbols like ▲ and *
    cleaned = re.sub(re.escape('▲'), '', text)
    cleaned = re.sub(re.escape('*'), '', cleaned)
    # Strip any remaining leading/trailing whitespace
    return cleaned.strip()


# Data
groups = [clean_label(key) for key in stat_metrics_sex_table_point.columns]
accuracy = stat_metrics_sex_table_point.loc[stat_metrics_sex_table_point.index.to_series() == "Accuracy",:].iloc[0].tolist()
effect_size = [stat_test_results_sex[key]["Accuracy"].effect_size if clean_label(key) in stat_test_results_sex.keys() else 0 for key in groups]

groups = [group.replace("_"," ") for group in groups]

x = np.arange(len(groups))
width = 0.35  # width of the bars

# Create grouped bar chart with dual axes to handle different scales properly
fig, ax1 = plt.subplots(figsize=(10, 6.0)) # Optimized size for horizontal legends

color1 = '#1f77b4' # Muted blue
rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.2)

# --- CRITICAL FIX: Removed x-axis label as requested ---
ax1.set_xlabel('', labelpad=0) 

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = '#ff7f0e' # Muted orange
rects2 = ax2.bar(x + width/2, effect_size, width, label='Effect Size', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Effect Size', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# --- CRITICAL FIX: Right axis limits set from 0 to 1 ---
ax2.set_ylim(0, 1.2) 

# --- CRITICAL FIX: Turn grid on ---
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars stay in front of grid lines
ax1.set_axisbelow(True)

# Add values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=15, ha='right', fontsize=11)

# --- CRITICAL FIX: Main Metrics Legend set to horizontal line (ncol=2) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
main_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, frameon=True, shadow=False)
ax1.add_artist(main_legend) # Freeze this legend layer to allow extra ones

# Configure the figure title
plt.title('Sex Group-Specific Accuracy and Bias Effect Size (Side-by-Side)', fontsize=13, fontweight='bold', pad=15)

# --- CRITICAL FIX: Stacked horizontal banners placed one below the other using fixed vertical coordinates ---
horizontal_legend_text_sign = r"$\bf{Significance\ Indicators:}$ $*$ Omnibus significance, $\blacktriangle$ Pairwise vs. ref. group"
horizontal_legend_text_eff_size = r"$\bf{Effect\ Size\ Thresholds:}$ Small $\leq$ 0.2, Medium $\leq$ 0.6, Large $>$ 0.6"

props_footer = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)

# Placed horizontally centered, stacked vertically near the bottom canvas window
# fig.text(0.544, -0.05, horizontal_legend_text_sign, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')
fig.text(0.5, 0.00, horizontal_legend_text_eff_size, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')

# Adjust layout to cleanly incorporate the bottom text legend without cutoff
plt.tight_layout()
plt.subplots_adjust(bottom=0.16) 

# plt.savefig('grouped_bar_chart_side_by_side.png', dpi=300)
# plt.close()

print("Grouped bar chart generated successfully.")


plt.savefig('images/bar_chart_accuracy_sex_effect.pdf', dpi=600, bbox_inches='tight')

# %%
# PR curves
eqb.eq_plot_group_curves(
    sliced_sex_data,
    curve_type="pr",
    subplots=False,
    figsize=(10, 8),
    title="Precision-Recall by Sex Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="pr_curves_sex.pdf"
)

# %%
# ROC curves
eqb.eq_plot_group_curves(
    sliced_sex_data,
    curve_type="roc",
    title="ROC AUC by Sex Group",
    figsize=(10, 8),
    decimal_places=2,
    subplots=False,
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="roc_curves_sex.pdf"
)

# %%
# calibration curves
eqb.eq_plot_group_curves(
    sliced_sex_data,
    curve_type="calibration",
    shade_area=True,
    figsize=(10,8),
    title="Calibration by Sex Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    subplots=False,
    save_path="images",
    filename="cal_curves_sex.pdf"
)


# %% [markdown]
# #### Ethnicity

# %%
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_label(text):
    # Remove specific symbols like ▲ and *
    cleaned = re.sub(re.escape('▲'), '', text)
    cleaned = re.sub(re.escape('*'), '', cleaned)
    # Strip any remaining leading/trailing whitespace
    return cleaned.strip()

# Data
groups = [clean_label(key) for key in stat_metrics_ethn_table_point.columns] 
accuracy = stat_metrics_ethn_table_point.loc[stat_metrics_ethn_table_point.index.to_series() == "Accuracy",:].iloc[0].tolist()
effect_size = [stat_test_results_ethn[key]["Accuracy"].effect_size if key in stat_test_results_ethn.keys() else 0 for key in groups]

groups = [group.replace("_"," ") for group in groups]

x = np.arange(len(groups))
width = 0.35  # width of the bars

# Create grouped bar chart with dual axes to handle different scales properly
fig, ax1 = plt.subplots(figsize=(10, 6.0)) # Optimized size for horizontal legends

color1 = '#1f77b4' # Muted blue
rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.05)

# --- CRITICAL FIX: Removed x-axis label as requested ---
ax1.set_xlabel('', labelpad=0) 

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = '#ff7f0e' # Muted orange
rects2 = ax2.bar(x + width/2, effect_size, width, label='Effect Size', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Effect Size', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# --- CRITICAL FIX: Right axis limits set from 0 to 1 ---
ax2.set_ylim(0, 1.00) 

# --- CRITICAL FIX: Turn grid on ---
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars stay in front of grid lines
ax1.set_axisbelow(True)

# Add values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=15, ha='right', fontsize=11)

# --- CRITICAL FIX: Main Metrics Legend set to horizontal line (ncol=2) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
main_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, frameon=True, shadow=False)
ax1.add_artist(main_legend) # Freeze this legend layer to allow extra ones

# Configure the figure title
plt.title('Ethnicity Group-Specific Accuracy and Bias Effect Size (Side-by-Side)', fontsize=13, fontweight='bold', pad=15)

# --- CRITICAL FIX: Stacked horizontal banners placed one below the other using fixed vertical coordinates ---
horizontal_legend_text_sign = r"$\bf{Significance\ Indicators:}$ $*$ Omnibus significance, $\blacktriangle$ Pairwise vs. ref. group"
horizontal_legend_text_eff_size = r"$\bf{Effect\ Size\ Thresholds:}$ Small $\leq$ 0.2, Medium $\leq$ 0.6, Large $>$ 0.6"

props_footer = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)

# Placed horizontally centered, stacked vertically near the bottom canvas window
# fig.text(0.544, -0.05, horizontal_legend_text_sign, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')
fig.text(0.5, 0.00, horizontal_legend_text_eff_size, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')

# Adjust layout to cleanly incorporate the bottom text legend without cutoff
plt.tight_layout()
plt.subplots_adjust(bottom=0.16) 

# plt.savefig('grouped_bar_chart_side_by_side.png', dpi=300)
# plt.close()

print("Grouped bar chart generated successfully.")

plt.savefig('images/bar_chart_accuracy_ethnicity_effect.pdf', dpi=600, bbox_inches='tight')

# %%
# PR curves
eqb.eq_plot_group_curves(
    sliced_ethn_data,
    curve_type="pr",
    subplots=False,
    figsize=(10, 8),
    title="Precision-Recall by Ethnicity Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="pr_curves_ethnicity.pdf"
)

# %%
# ROC curves
eqb.eq_plot_group_curves(
    sliced_ethn_data,
    curve_type="roc",
    title="ROC AUC by Ethnicity Group",
    figsize=(10, 8),
    decimal_places=2,
    subplots=False,
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="roc_curves_ethnicity.pdf"
)

# %%
# calibration curves
eqb.eq_plot_group_curves(
    sliced_ethn_data,
    curve_type="calibration",
    shade_area=True,
    figsize=(10,8),
    title="Calibration by Ethnicity Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    subplots=False,
    save_path="images",
    filename="cal_curves_ethnicity.pdf"
)

# %% [markdown]
# #### Age

# %%
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_label(text):
    # Remove specific symbols like ▲ and *
    cleaned = re.sub(re.escape('▲'), '', text)
    cleaned = re.sub(re.escape('*'), '', cleaned)
    # Strip any remaining leading/trailing whitespace
    return cleaned.strip()

# Data
groups = [clean_label(key) for key in stat_metrics_age_table_point.columns] 
accuracy = stat_metrics_age_table_point.loc[stat_metrics_age_table_point.index.to_series() == "Accuracy",:].iloc[0].tolist()
effect_size = [stat_test_results_age[key]["Accuracy"].effect_size if key in stat_test_results_age.keys() else 0 for key in groups]

groups = [group.replace("_"," ") for group in groups]

x = np.arange(len(groups))
width = 0.35  # width of the bars

# Create grouped bar chart with dual axes to handle different scales properly
fig, ax1 = plt.subplots(figsize=(10, 6.0)) # Optimized size for horizontal legends

color1 = '#1f77b4' # Muted blue
rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.2)

# --- CRITICAL FIX: Removed x-axis label as requested ---
ax1.set_xlabel('', labelpad=0) 

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = '#ff7f0e' # Muted orange
rects2 = ax2.bar(x + width/2, effect_size, width, label='Effect Size', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Effect Size', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# --- CRITICAL FIX: Right axis limits set from 0 to 1 ---
ax2.set_ylim(0, 1.2) 

# --- CRITICAL FIX: Turn grid on ---
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars stay in front of grid lines
ax1.set_axisbelow(True)

# Add values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=15, ha='right', fontsize=11)

# --- CRITICAL FIX: Main Metrics Legend set to horizontal line (ncol=2) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
main_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, frameon=True, shadow=False)
ax1.add_artist(main_legend) # Freeze this legend layer to allow extra ones

# Configure the figure title
plt.title('Age Group-Specific Accuracy and Bias Effect Size (Side-by-Side)', fontsize=13, fontweight='bold', pad=15)

# --- CRITICAL FIX: Stacked horizontal banners placed one below the other using fixed vertical coordinates ---
horizontal_legend_text_sign = r"$\bf{Significance\ Indicators:}$ $*$ Omnibus significance, $\blacktriangle$ Pairwise vs. ref. group"
horizontal_legend_text_eff_size = r"$\bf{Effect\ Size\ Thresholds:}$ Small $\leq$ 0.2, Medium $\leq$ 0.6, Large $>$ 0.6"

props_footer = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)

# Placed horizontally centered, stacked vertically near the bottom canvas window
# fig.text(0.544, -0.05, horizontal_legend_text_sign, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')
fig.text(0.5, 0.00, horizontal_legend_text_eff_size, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')

# Adjust layout to cleanly incorporate the bottom text legend without cutoff
plt.tight_layout()
plt.subplots_adjust(bottom=0.16) 

# plt.savefig('grouped_bar_chart_side_by_side.png', dpi=300)
# plt.close()

print("Grouped bar chart generated successfully.")

plt.savefig('images/bar_chart_accuracy_age_effect.pdf', dpi=600, bbox_inches='tight')

# %%
# PR curves
eqb.eq_plot_group_curves(
    sliced_age_data,
    curve_type="pr",
    subplots=False,
    figsize=(10, 8),
    title="Precision-Recall by Age Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="pr_curves_age.pdf"
)

# %%
# ROC curves
eqb.eq_plot_group_curves(
    sliced_age_data,
    curve_type="roc",
    title="ROC AUC by Age Group",
    figsize=(10, 8),
    decimal_places=2,
    subplots=False,
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="roc_curves_age.pdf"
)

# %%
# calibration curves
eqb.eq_plot_group_curves(
    sliced_age_data,
    curve_type="calibration",
    shade_area=True,
    figsize=(10,8),
    title="Calibration by Age Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    subplots=False,
    save_path="images",
    filename="cal_curves_age.pdf"
)

# %% [markdown]
# #### SVI

# %%
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_label(text):
    # Remove specific symbols like ▲ and *
    cleaned = re.sub(re.escape('▲'), '', text)
    cleaned = re.sub(re.escape('*'), '', cleaned)
    # Strip any remaining leading/trailing whitespace
    return cleaned.strip()

# Data
groups = [clean_label(key) for key in stat_metrics_svi_table_point.columns] 
accuracy = stat_metrics_svi_table_point.loc[stat_metrics_svi_table_point.index.to_series() == "Accuracy",:].iloc[0].tolist()
effect_size = [stat_test_results_svi[key]["Accuracy"].effect_size if key in stat_test_results_svi.keys() else 0 for key in groups]

groups = [group.replace("_"," ") for group in groups]

x = np.arange(len(groups))
width = 0.35  # width of the bars

# Create grouped bar chart with dual axes to handle different scales properly
fig, ax1 = plt.subplots(figsize=(10, 6.0)) # Optimized size for horizontal legends

color1 = '#1f77b4' # Muted blue
rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color1, edgecolor='black', alpha=0.85)
ax1.set_ylabel('Accuracy', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.2)

# --- CRITICAL FIX: Removed x-axis label as requested ---
ax1.set_xlabel('', labelpad=0) 

# Instantiate a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = '#ff7f0e' # Muted orange
rects2 = ax2.bar(x + width/2, effect_size, width, label='Effect Size', color=color2, edgecolor='black', alpha=0.85)
ax2.set_ylabel('Effect Size', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)

# --- CRITICAL FIX: Right axis limits set from 0 to 1 ---
ax2.set_ylim(0, 1.2) 

# --- CRITICAL FIX: Turn grid on ---
ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5, zorder=0)
# Ensure bars stay in front of grid lines
ax1.set_axisbelow(True)

# Add values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:.3f}',
                 xy=(rect.get_x() + rect.get_width()/2, h),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Set x-ticks
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=15, ha='right', fontsize=11)

# --- CRITICAL FIX: Main Metrics Legend set to horizontal line (ncol=2) ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
main_legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=2, frameon=True, shadow=False)
ax1.add_artist(main_legend) # Freeze this legend layer to allow extra ones

# Configure the figure title
plt.title('SVI Group-Specific Accuracy and Bias Effect Size (Side-by-Side)', fontsize=13, fontweight='bold', pad=15)

# --- CRITICAL FIX: Stacked horizontal banners placed one below the other using fixed vertical coordinates ---
horizontal_legend_text_sign = r"$\bf{Significance\ Indicators:}$ $*$ Omnibus significance, $\blacktriangle$ Pairwise vs. ref. group"
horizontal_legend_text_eff_size = r"$\bf{Effect\ Size\ Thresholds:}$ Small $\leq$ 0.2, Medium $\leq$ 0.6, Large $>$ 0.6"

props_footer = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.85)

# Placed horizontally centered, stacked vertically near the bottom canvas window
# fig.text(0.544, -0.05, horizontal_legend_text_sign, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')
fig.text(0.5, 0.00, horizontal_legend_text_eff_size, fontsize=9.5, bbox=props_footer, transform=fig.transFigure, ha='right', va='bottom')

# Adjust layout to cleanly incorporate the bottom text legend without cutoff
plt.tight_layout()
plt.subplots_adjust(bottom=0.16) 

# plt.savefig('grouped_bar_chart_side_by_side.png', dpi=300)
# plt.close()

print("Grouped bar chart generated successfully.")

plt.savefig('images/bar_chart_accuracy_svi_effect.pdf', dpi=600, bbox_inches='tight')

# %%
# PR curves
eqb.eq_plot_group_curves(
    sliced_svi_data,
    curve_type="pr",
    subplots=False,
    figsize=(10, 8),
    title="Precision-Recall by SVI Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="pr_curves_svi.pdf"
)

# %%
# ROC curves
eqb.eq_plot_group_curves(
    sliced_svi_data,
    curve_type="roc",
    title="ROC AUC by SVI Group",
    figsize=(10, 8),
    decimal_places=2,
    subplots=False,
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    save_path="images",
    filename="roc_curves_svi.pdf"
)

# %%
# calibration curves
eqb.eq_plot_group_curves(
    sliced_svi_data,
    curve_type="calibration",
    shade_area=True,
    figsize=(10,8),
    title="Calibration by SVI Group",
    exclude_groups=["Amer-Indian-Eskimo", "Other"],
    subplots=False,
    save_path="images",
    filename="cal_curves_svi.pdf"
)

# %% [markdown]
# ## Bootstrap

# %%
# setting fixed seed for reproducibility
# Alternatively, seeds can be set after initialization
int_list = np.linspace(0, len(y_true), num=len(y_true), dtype=int).tolist()

eq2 = eqb.EquiBoots(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    fairness_df=fairness_df,
    # fairness_vars=["Race", "Sex", "Ethnicity", "SexualOrientation", "age_group", "SVI_category", "eGFR_stage"],
    # fairness_vars=["Race", "Sex", "Ethnicity", "age_group", "SVI_category",],
    # fairness_vars=["Ethnicity"],
    fairness_vars=["Race", "Sex", "Ethnicity", "age_group", "SVI_category", "eGFR_stage"],
    seeds=int_list,
    # reference_groups=["white", "Male", "Non-Hispanic", "Straight", "Older Adult", "Low Vulnerability", "Stage 2 (Mildly Decreased)"],
    # reference_groups=["white", "Male", "Non-Hispanic", "Older Adult", "Low Vulnerability",],
    # reference_groups=["Non-Hispanic"],
    reference_groups=["white", "Male", "Non-Hispanic", "Older Adult", "Low Vulnerability", "Stage 2 (Mildly Decreased)"],
    task="binary_classification",
    bootstrap_flag=True,
    num_bootstraps=5001,
    # num_bootstraps=50, # debug purposes
    boot_sample_size=len(y_true),  # whole length of test set
    # group_min_size=150,  # any group with samples below this number will be ignored
    balanced=False,  # False is stratified (i.e., maintaining groups proportions), True is balanced (equal proportions)
    stratify_by_outcome=True,  # True maintain initial dataset outcome proportions per group
    group_min_size=10
)

# Set seeds after initialization
eq2.set_fix_seeds(int_list)
print("seeds", eq2.seeds)



# %%
# group bootstraps by grouping variables (e.g., race)
# eq2.grouper(groupings_vars=["Race", "Sex", "Ethnicity", "SexualOrientation", "age_group", "SVI_category", "eGFR_stage"])
# eq2.grouper(groupings_vars=["Race", "Sex", "Ethnicity", "age_group", "SVI_category"])
# eq2.grouper(groupings_vars=["Ethnicity"])
eq2.grouper(groupings_vars=["Race", "Sex", "Ethnicity", "age_group", "SVI_category", "eGFR_stage"])

# %%
# slice by variable and assign to a variable
# race related bootstraps
boots_race_data = eq2.slicer("Race")
boots_sex_data = eq2.slicer("Sex")
boots_ethn_data = eq2.slicer("Ethnicity")
# boots_sex_or_data = eq2.slicer("SexualOrientation")



# compute binary classification metrics wrt to race
boots_race_metrics = eq2.get_metrics(boots_race_data)
boots_sex_metrics = eq2.get_metrics(boots_sex_data)
boots_ethn_metrics = eq2.get_metrics(boots_ethn_data)
# boots_sex_or_metrics = eq2.get_metrics(boots_sex_or_data)


# %%
dispa_race = eq2.calculate_disparities(boots_race_metrics, "Race")
dispa_sex = eq2.calculate_disparities(boots_sex_metrics, "Sex")
dispa_ethn = eq2.calculate_disparities(boots_ethn_metrics, "Ethnicity")
# dispa_sex_or = eq2.calculate_disparities(boots_sex_or_metrics, "SexualOrientation")


# %% [markdown]
# ## Disparity Ratios

# %%
eqb.eq_group_metrics_plot(
    group_metrics=dispa_race,
    metric_cols=[
        "Accuracy_Ratio",
        "Precision_Ratio",
        "Predicted_Prevalence_Ratio",
        "Prevalence_Ratio",
        "FP_Rate_Ratio",
        "TN_Rate_Ratio",
        "Recall_Ratio",
    ],
    name="race",
    categories="all",
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    strict_layout=True,
    leg_cols=7,
    figsize=(35,6),
    # plot_thresholds=[0.9, 1.2],
    save_path="images",
    filename="disp_ratios_race.pdf"
)

# %%
eqb.eq_group_metrics_plot(
    group_metrics=dispa_sex,
    metric_cols=[
        "Accuracy_Ratio",
        "Precision_Ratio",
        "Predicted_Prevalence_Ratio",
        "Prevalence_Ratio",
        "FP_Rate_Ratio",
        "TN_Rate_Ratio",
        "Recall_Ratio",
    ],
    name="Sex",
    categories="all",
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    strict_layout=True,
    leg_cols=7,
    # plot_thresholds=[0.9, 1.2],
    figsize=(35,6),
    # plot_thresholds=[0.9, 1.2],
    save_path="images",
    filename="disp_ratios_sex.pdf"
)

# %%
eqb.eq_group_metrics_plot(
    group_metrics=dispa_ethn,
    metric_cols=[
        "Accuracy_Ratio",
        "Precision_Ratio",
        "Predicted_Prevalence_Ratio",
        "Prevalence_Ratio",
        "FP_Rate_Ratio",
        "TN_Rate_Ratio",
        "Recall_Ratio",
    ],
    name="Ethnicity",
    categories="all",
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    strict_layout=True,
    leg_cols=7,
    # plot_thresholds=[0.9, 1.2],
    figsize=(35,6),
    # plot_thresholds=[0.9, 1.2],
    save_path="images",
    filename="disp_ratios_ethnicity.pdf"
)


# %%
diffs_race = eq2.calculate_differences(boots_race_metrics, "Race")
diffs_sex = eq2.calculate_differences(boots_sex_metrics, "Sex")
diffs_ethn = eq2.calculate_differences(boots_ethn_metrics, "Ethnicity")
# diffs_sex_or = eq2.calculate_differences(boots_sex_or_metrics, "SexualOrientation")


# %%
# metrics to perform a statistical test
metrics_boot = [
    "Accuracy_diff",
    "Precision_diff",
    "Recall_diff",
    "F1_Score_diff",
    "Specificity_diff",
    "TP_Rate_diff",
    "FP_Rate_diff",
    "FN_Rate_diff",
    "TN_Rate_diff",
    "Prevalence_diff",
    "Predicted_Prevalence_diff",
    "ROC_AUC_diff",
    "Average_Precision_Score_diff",
    "Log_Loss_diff",
    "Brier_Score_diff",
    "Calibration_AUC_diff",
]

# %% [markdown]
# ## Differences
# 
# ## Race

# %%
# configuration dictionary to provide parameters around statistical testing
test_config = {
    "test_type": "bootstrap_test",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
    "tail_type": "two_tailed",
    "metrics": metrics_boot,
}


stat_test_results = eq2.analyze_statistical_significance(
    metric_dict=boots_race_metrics,  # pass variable sliced metrics
    var_name="Race",  # variable name
    test_config=test_config,  # configuration
    differences=diffs_race,  # the differences of each race group
)

stat_metrics_table_diff = metrics_table(
    boots_race_metrics,
    statistical_tests=stat_test_results,
    differences=diffs_race,
    reference_group="White",
)

###### set reference group in plot
var_name = "Race"

ref = eq2.reference_groups[var_name]

for row in diffs_race:
    row[ref] = {col: 0.0 for col in metrics_boot}

# differences of each race group wrt reference group
# reference group differences are all zero not shown for simplicity
# * depicts statistical significance
# stat_metrics_table_diff

eqb.eq_group_metrics_plot(
    group_metrics=diffs_race,
    metric_cols=metrics_boot,
    name="race",
    categories="all",
    figsize=(30, 12),
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    max_cols=6,
    strict_layout=True,
    show_pass_fail=False,
    statistical_tests=stat_test_results,
    disparities=True,
    include_legend=True,
    y_lim=(-0.5, 0.5),
    save_path="images",
    filename="disp_diffs_race.pdf"
)

# %%
eqb.calculate_bootstrap_stats(group_boot_metrics=boots_race_metrics, metric="ROC AUC")

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_race_data,
    curve_type="roc",
    title="Bootstrapped ROC Curve by Race",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_roc_curv_race.pdf"
)

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_race_data,
    curve_type="pr",
    title="Bootstrapped PR Curve by Race",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_pr_curv_race.pdf"
)

# %% [markdown]
# ## Sex

# %%
# metrics to perform a statistical test
metrics_boot = [
    "Accuracy_diff",
    "Precision_diff",
    "Recall_diff",
    "F1_Score_diff",
    "Specificity_diff",
    "TP_Rate_diff",
    "FP_Rate_diff",
    "FN_Rate_diff",
    "TN_Rate_diff",
    "Prevalence_diff",
    "Predicted_Prevalence_diff",
    "ROC_AUC_diff",
    "Average_Precision_Score_diff",
    "Log_Loss_diff",
    "Brier_Score_diff",
    "Calibration_AUC_diff",
]

# configuration dictionary to provide parameters around statistical testing
test_config = {
    "test_type": "bootstrap_test",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
    "tail_type": "two_tailed",
    "metrics": metrics_boot,
}


stat_test_results = eq2.analyze_statistical_significance(
    metric_dict=boots_sex_metrics,  # pass variable sliced metrics
    var_name="Sex",  # variable name
    test_config=test_config,  # configuration
    differences=diffs_sex,  # the differences of each race group
)

stat_metrics_table_diff = metrics_table(
    boots_sex_metrics,
    statistical_tests=stat_test_results,
    differences=diffs_sex,
    reference_group="Male",
)

###### set reference group in plot
var_name = "Sex"
ref = eq2.reference_groups[var_name]

for row in diffs_sex:
    row[ref] = {col: 0.0 for col in metrics_boot}

# differences of each race group wrt reference group
# reference group differences are all zero not shown for simplicity
# * depicts statistical significance
# stat_metrics_table_diff

eqb.eq_group_metrics_plot(
    group_metrics=diffs_sex,
    metric_cols=metrics_boot,
    name="Sex",
    categories="all",
    figsize=(30, 12),
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    max_cols=6,
    strict_layout=True,
    show_pass_fail=False,
    statistical_tests=stat_test_results,
    disparities=True,
    y_lim=(-.5, 0.5),
    save_path="images",
    filename="disp_diffs_sex.pdf"
)

# %%
eqb.calculate_bootstrap_stats(group_boot_metrics=boots_sex_metrics, metric="ROC AUC")

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_sex_data,
    curve_type="roc",
    title="Bootstrapped ROC Curve by Sex",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_roc_curv_sex.pdf"
)

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_sex_data,
    curve_type="pr",
    title="Bootstrapped PR Curve by Sex",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_pr_curv_sex.pdf"
)

# %% [markdown]
# ## Ethnicity

# %%
# metrics to perform a statistical test
metrics_boot = [
    "Accuracy_diff",
    "Precision_diff",
    "Recall_diff",
    "F1_Score_diff",
    "Specificity_diff",
    "TP_Rate_diff",
    "FP_Rate_diff",
    "FN_Rate_diff",
    "TN_Rate_diff",
    "Prevalence_diff",
    "Predicted_Prevalence_diff",
    "ROC_AUC_diff",
    "Average_Precision_Score_diff",
    "Log_Loss_diff",
    "Brier_Score_diff",
    "Calibration_AUC_diff",
]

# configuration dictionary to provide parameters around statistical testing
test_config = {
    "test_type": "bootstrap_test",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
    "tail_type": "two_tailed",
    "metrics": metrics_boot,
}


stat_test_results = eq2.analyze_statistical_significance(
    metric_dict=boots_ethn_metrics,  # pass variable sliced metrics
    var_name="Ethnicity",  # variable name
    test_config=test_config,  # configuration
    differences=diffs_ethn,  # the differences of each race group
)

stat_metrics_table_diff = metrics_table(
    boots_ethn_metrics,
    statistical_tests=stat_test_results,
    differences=diffs_ethn,
    reference_group="Choose Not to Answer",
)


###### set reference group in plot
var_name = "Ethnicity"

ref = eq2.reference_groups[var_name]

for row in diffs_ethn:
    row[ref] = {col: 0.0 for col in metrics_boot}

# differences of each race group wrt reference group
# reference group differences are all zero not shown for simplicity
# * depicts statistical significance
# stat_metrics_table_diff

eqb.eq_group_metrics_plot(
    group_metrics=diffs_ethn,
    metric_cols=metrics_boot,
    name="Ethnicity",
    categories="all",
    figsize=(30, 12),
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    max_cols=6,
    strict_layout=True,
    show_pass_fail=False,
    statistical_tests=stat_test_results,
    disparities=True,
    y_lim=(-0.5, 0.5),
    save_path="images",
    filename="disp_diffs_ethnicity.pdf"
)

# %%
eqb.calculate_bootstrap_stats(group_boot_metrics=boots_ethn_metrics, metric="ROC AUC")

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_ethn_data,
    curve_type="roc",
    title="Bootstrapped ROC Curve by Ethnicity",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_roc_curv_ethnicity.pdf"
)

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_ethn_data,
    curve_type="pr",
    title="Bootstrapped PR Curve by Ethnicity",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_pr_curv_ethnicity.pdf"
)

# %% [markdown]
# ## SexualOrientation

# %%
# # metrics to perform a statistical test
# metrics_boot = [
#     "Accuracy_diff",
#     "Precision_diff",
#     "Recall_diff",
#     "F1_Score_diff",
#     "Specificity_diff",
#     "TP_Rate_diff",
#     "FP_Rate_diff",
#     "FN_Rate_diff",
#     "TN_Rate_diff",
#     "Prevalence_diff",
#     "Predicted_Prevalence_diff",
#     "ROC_AUC_diff",
#     "Average_Precision_Score_diff",
#     "Log_Loss_diff",
#     "Brier_Score_diff",
#     "Calibration_AUC_diff",
# ]

# # configuration dictionary to provide parameters around statistical testing
# test_config = {
#     "test_type": "bootstrap_test",
#     "alpha": 0.05,
#     "adjust_method": "bonferroni",
#     "confidence_level": 0.95,
#     "classification_task": "binary_classification",
#     "tail_type": "two_tailed",
#     "metrics": metrics_boot,
# }


# stat_test_results = eq2.analyze_statistical_significance(
#     metric_dict=boots_sex_or_metrics,  # pass variable sliced metrics
#     var_name="SexualOrientation",  # variable name
#     test_config=test_config,  # configuration
#     differences=diffs_sex_or,  # the differences of each race group
# )

# stat_metrics_table_diff = metrics_table(
#     boots_sex_or_metrics,
#     statistical_tests=stat_test_results,
#     differences=diffs_sex_or,
#     reference_group="Choose Not to Answer",
# )

# ###### set reference group in plot
# var_name = "SexualOrientation"

# ref = eq2.reference_groups[var_name]

# for row in diffs_sex_or:
#     row[ref] = {col: 0.0 for col in metrics_boot}

# # differences of each race group wrt reference group
# # reference group differences are all zero not shown for simplicity
# # * depicts statistical significance
# # stat_metrics_table_diff

# eqb.eq_group_metrics_plot(
#     group_metrics=diffs_sex_or,
#     metric_cols=metrics_boot,
#     name="SexualOrientation",
#     categories="all",
#     figsize=(20, 10),
#     plot_type="violinplot",
#     color_by_group=True,
#     show_grid=True,
#     max_cols=6,
#     strict_layout=True,
#     save_path="./images",
#     show_pass_fail=False,
#     statistical_tests=stat_test_results,
#     disparities=True,
#     y_lim=(-0.5, 0.5),
# )

# %%
# eqb.calculate_bootstrap_stats(group_boot_metrics=boots_sex_or_metrics, metric="ROC AUC")

# %%
# eqb.eq_plot_bootstrapped_group_curves(
#     boot_sliced_data=boots_sex_or_data,
#     curve_type="roc",
#     title="Bootstrapped ROC Curve by Sexual Orientation",
#     bar_every=100,
#     subplots=True,
#     dpi=300,
#     n_bins=10,
#     figsize=(6, 4),
#     color_by_group=True,
# )

# %%
# eqb.eq_plot_bootstrapped_group_curves(
#     boot_sliced_data=boots_sex_or_data,
#     curve_type="pr",
#     title="Bootstrapped PR Curve by Sexual Orientation",
#     bar_every=100,
#     subplots=True,
#     dpi=300,
#     n_bins=10,
#     figsize=(6, 4),
#     color_by_group=True,
# )

# %% [markdown]
# ### Clearing memory

# %%
import gc

# 1. Unbind the metrics and sliced data variables
del (
    boots_race_data,
    boots_sex_data,
    boots_ethn_data,
    boots_race_metrics,
    boots_sex_metrics,
    boots_ethn_metrics,
)

# 2. Force the garbage collector to free the memory
gc.collect()

# %%
boots_age_gr_data = eq2.slicer("age_group")
boots_svi_data = eq2.slicer("SVI_category")
boots_egfr_data = eq2.slicer("eGFR_stage")


boots_age_gr_metrics = eq2.get_metrics(boots_age_gr_data)
boots_svi_metrics = eq2.get_metrics(boots_svi_data)
boots_egfr_metrics = eq2.get_metrics(boots_egfr_data)

# %%
dispa_age_gr = eq2.calculate_disparities(boots_age_gr_metrics, "age_group")
dispa_svi = eq2.calculate_disparities(boots_svi_metrics, "SVI_category")
dispa_egfr = eq2.calculate_disparities(boots_egfr_metrics, "eGFR_stage")

# %%

eqb.eq_group_metrics_plot(
    group_metrics=dispa_age_gr,
    metric_cols=[
        "Accuracy_Ratio",
        "Precision_Ratio",
        "Predicted_Prevalence_Ratio",
        "Prevalence_Ratio",
        "FP_Rate_Ratio",
        "TN_Rate_Ratio",
        "Recall_Ratio",
    ],
    name="Age",
    categories="all",
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    strict_layout=True,
    leg_cols=7,
    # plot_thresholds=[0.9, 1.2],
    figsize=(35,6),
    # plot_thresholds=[0.9, 1.2],
    save_path="images",
    filename="disp_ratios_age.pdf"
)

# %%

eqb.eq_group_metrics_plot(
    group_metrics=dispa_svi,
    metric_cols=[
        "Accuracy_Ratio",
        "Precision_Ratio",
        "Predicted_Prevalence_Ratio",
        "Prevalence_Ratio",
        "FP_Rate_Ratio",
        "TN_Rate_Ratio",
        "Recall_Ratio",
    ],
    name="SVI",
    categories="all",
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    # disparities=True,
    strict_layout=True,
    leg_cols=7,
    # plot_thresholds=[0.9, 1.2],
    figsize=(35,6),
    # plot_thresholds=[0.9, 1.2],
    save_path="images",
    filename="disp_ratios_svi.pdf"
)

# %%

eqb.eq_group_metrics_plot(
    group_metrics=dispa_egfr,
    metric_cols=[
        "Accuracy_Ratio",
        "Precision_Ratio",
        "Predicted_Prevalence_Ratio",
        "Prevalence_Ratio",
        "FP_Rate_Ratio",
        "TN_Rate_Ratio",
        "Recall_Ratio",
    ],
    name="eGFR Stage",
    categories="all",
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    strict_layout=True,
    leg_cols=7,
    # disparities=False
    # plot_thresholds=[0.9, 1.2],
    figsize=(35,6),
    # plot_thresholds=[0.9, 1.2],
    save_path="images",
    filename="disp_ratios_egfr.pdf"
)

# %%
diffs_age_gr = eq2.calculate_differences(boots_age_gr_metrics, "age_group")
diffs_svi = eq2.calculate_differences(boots_svi_metrics, "SVI_category")
diffs_egfr = eq2.calculate_differences(boots_egfr_metrics, "eGFR_stage")

# %% [markdown]
# ## Age

# %%
# metrics to perform a statistical test
metrics_boot = [
    "Accuracy_diff",
    "Precision_diff",
    "Recall_diff",
    "F1_Score_diff",
    "Specificity_diff",
    "TP_Rate_diff",
    "FP_Rate_diff",
    "FN_Rate_diff",
    "TN_Rate_diff",
    "Prevalence_diff",
    "Predicted_Prevalence_diff",
    "ROC_AUC_diff",
    "Average_Precision_Score_diff",
    "Log_Loss_diff",
    "Brier_Score_diff",
    "Calibration_AUC_diff",
]

# configuration dictionary to provide parameters around statistical testing
test_config = {
    "test_type": "bootstrap_test",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
    "tail_type": "two_tailed",
    "metrics": metrics_boot,
}


stat_test_results = eq2.analyze_statistical_significance(
    metric_dict=boots_age_gr_metrics,  # pass variable sliced metrics
    var_name="age_group",  # variable name
    test_config=test_config,  # configuration
    differences=diffs_age_gr,  # the differences of each race group
)

stat_metrics_table_diff = metrics_table(
    boots_age_gr_metrics,
    statistical_tests=stat_test_results,
    differences=diffs_age_gr,
    reference_group="Older Adult",
)

###### set reference group in plot
var_name = "age_group"

ref = eq2.reference_groups[var_name]

for row in diffs_age_gr:
    row[ref] = {col: 0.0 for col in metrics_boot}

# differences of each race group wrt reference group
# reference group differences are all zero not shown for simplicity
# * depicts statistical significance
# stat_metrics_table_diff

eqb.eq_group_metrics_plot(
    group_metrics=diffs_age_gr,
    metric_cols=metrics_boot,
    name="age_group",
    categories="all",
    figsize=(30, 12),
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    max_cols=6,
    strict_layout=True,
    show_pass_fail=False,
    statistical_tests=stat_test_results,
    disparities=True,
    y_lim=(-0.5, 0.5),
    save_path="images",
    filename="disp_diffs_age.pdf"
)

# %%
eqb.calculate_bootstrap_stats(group_boot_metrics=boots_age_gr_metrics, metric="ROC AUC")

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_age_gr_data,
    curve_type="roc",
    title="Bootstrapped ROC Curve by Age",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_roc_curv_age.pdf"
)

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_age_gr_data,
    curve_type="pr",
    title="Bootstrapped PR Curve by Age",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_pr_curv_age.pdf"
)

# %% [markdown]
# ## SVI

# %%
# metrics to perform a statistical test
metrics_boot = [
    "Accuracy_diff",
    "Precision_diff",
    "Recall_diff",
    "F1_Score_diff",
    "Specificity_diff",
    "TP_Rate_diff",
    "FP_Rate_diff",
    "FN_Rate_diff",
    "TN_Rate_diff",
    "Prevalence_diff",
    "Predicted_Prevalence_diff",
    "ROC_AUC_diff",
    "Average_Precision_Score_diff",
    "Log_Loss_diff",
    "Brier_Score_diff",
    "Calibration_AUC_diff",
]

# configuration dictionary to provide parameters around statistical testing
test_config = {
    "test_type": "bootstrap_test",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
    "tail_type": "two_tailed",
    "metrics": metrics_boot,
}


stat_test_results = eq2.analyze_statistical_significance(
    metric_dict=boots_svi_metrics,  # pass variable sliced metrics
    var_name="SVI_category",  # variable name
    test_config=test_config,  # configuration
    differences=diffs_svi,  # the differences of each race group
)

stat_metrics_table_diff = metrics_table(
    boots_svi_metrics,
    statistical_tests=stat_test_results,
    differences=diffs_svi,
    reference_group="Low Vulnerability",
)

###### set reference group in plot
var_name = "SVI_category"

ref = eq2.reference_groups[var_name]

for row in diffs_svi:
    row[ref] = {col: 0.0 for col in metrics_boot}

# differences of each race group wrt reference group
# reference group differences are all zero not shown for simplicity
# * depicts statistical significance
# stat_metrics_table_diff

eqb.eq_group_metrics_plot(
    group_metrics=diffs_svi,
    metric_cols=metrics_boot,
    name="SVI_category",
    categories="all",
    figsize=(30, 12),
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    max_cols=6,
    strict_layout=True,
    show_pass_fail=False,
    statistical_tests=stat_test_results,
    disparities=True,
    y_lim=(-0.5, 0.5),
    save_path="images",
    filename="disp_diffs_svi.pdf"
)

# %%
eqb.calculate_bootstrap_stats(group_boot_metrics=boots_svi_metrics, metric="ROC AUC")

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_svi_data,
    curve_type="roc",
    title="Bootstrapped ROC Curve by SVI",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_roc_curv_svi.pdf"
)

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_svi_data,
    curve_type="pr",
    title="Bootstrapped PR Curve by SVI",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_pr_curv_svi.pdf"
)

# %% [markdown]
# ## eGFR

# %%
# metrics to perform a statistical test
metrics_boot = [
    "Accuracy_diff",
    "Precision_diff",
    "Recall_diff",
    "F1_Score_diff",
    "Specificity_diff",
    "TP_Rate_diff",
    "FP_Rate_diff",
    "FN_Rate_diff",
    "TN_Rate_diff",
    "Prevalence_diff",
    "Predicted_Prevalence_diff",
    "ROC_AUC_diff",
    "Average_Precision_Score_diff",
    "Log_Loss_diff",
    "Brier_Score_diff",
    "Calibration_AUC_diff",
]

# configuration dictionary to provide parameters around statistical testing
test_config = {
    "test_type": "bootstrap_test",
    "alpha": 0.05,
    "adjust_method": "bonferroni",
    "confidence_level": 0.95,
    "classification_task": "binary_classification",
    "tail_type": "two_tailed",
    "metrics": metrics_boot,
}


stat_test_results = eq2.analyze_statistical_significance(
    metric_dict=boots_egfr_metrics,  # pass variable sliced metrics
    var_name="eGFR_stage",  # variable name
    test_config=test_config,  # configuration
    differences=diffs_egfr,  # the differences of each race group
)

stat_metrics_table_diff = metrics_table(
    boots_egfr_metrics,
    statistical_tests=stat_test_results,
    differences=diffs_egfr,
    reference_group="Stage 2 (Mildly Decreased)",
)

###### set reference group in plot
var_name = "eGFR_stage"

ref = eq2.reference_groups[var_name]

for row in diffs_egfr:
    row[ref] = {col: 0.0 for col in metrics_boot}

# differences of each race group wrt reference group
# reference group differences are all zero not shown for simplicity
# * depicts statistical significance
# stat_metrics_table_diff

eqb.eq_group_metrics_plot(
    group_metrics=diffs_egfr,
    metric_cols=metrics_boot,
    name="eGFR_stage",
    categories="all",
    figsize=(30, 12),
    plot_type="violinplot",
    color_by_group=True,
    show_grid=True,
    max_cols=6,
    strict_layout=True,
    show_pass_fail=False,
    statistical_tests=stat_test_results,
    disparities=True,
    y_lim=(-0.5, 0.5),
    save_path="images",
    filename="disp_diffs_egfr.pdf"
)

# %%
eqb.calculate_bootstrap_stats(group_boot_metrics=boots_egfr_metrics, metric="ROC AUC")

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_egfr_data,
    curve_type="roc",
    title="Bootstrapped ROC Curve by eGFR",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_roc_curv_egfr.pdf"
)

# %%
eqb.eq_plot_bootstrapped_group_curves(
    boot_sliced_data=boots_egfr_data,
    curve_type="pr",
    title="Bootstrapped PR Curve by eGFR",
    bar_every=100,
    subplots=True,
    dpi=300,
    n_bins=10,
    color_by_group=True,
    figsize=(12, 8),
    save_path="images",
    filename="boots_pr_curv_egfr.pdf"
)

# %%


# %%



