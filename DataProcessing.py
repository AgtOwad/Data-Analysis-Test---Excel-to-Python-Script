import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# Convert rank to an ordinal string (e.g., 1 -> "1st")
def ordinal(n):
    if pd.isna(n):
        return ""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    
# Custom formatter to display numbers in USD with "K" and "M" suffixes.
def usd_formatter(x, pos):
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    elif x >= 1e3:
        return f"${x/1e3:.0f}K"
    else:
        return f"${x:.0f}"


## --- Loading files ----
## "CENSUS_POPULATION_STATE.xlsx" and "CENSUS_MHI_STATE.xlsx", was loaded in as '.xlsx' in order to retain the formats of the file
## If loaded in as '.csv' it introduces new characters for " "
keys_df = pd.read_csv("KEYS.csv")
census_pop_df = pd.read_excel("CENSUS_POPULATION_STATE.xlsx")
census_mhi_df = pd.read_excel("CENSUS_MHI_STATE.xlsx")
redfin_df = pd.read_csv("REDFIN_MEDIAN_SALE_PRICE.csv", header=None)


# --- Prepare the keys dataset ---
# Rename relevant KEYS file columns names
keys_df.columns = ["key", "col2", "code", "code2", "col5", "type", "col7", "blurb"]

# Based on the Excel formulas, I filtered KEYS data to only "state"
keys_filtered = keys_df[(keys_df["type"] == "state") & (~keys_df["key"].str.contains("'"))].copy()

## This is where the final result will be stored
results = []

# Determine the last non-empty column from the Redfin header row
# MEDIAN SALES Data row 1 holds the header information (dates)
redfin_header = redfin_df.iloc[1]
last_valid_col = redfin_header.last_valid_index()

# Convert the header value to a datetime object for formatting.
try:
    redfin_date = pd.to_datetime(redfin_header[last_valid_col])
    redfin_date_str = redfin_date.strftime("%B %Y")
except Exception:
    redfin_date_str = str(redfin_header[last_valid_col])

# For each key (state) in the data we clean the Census Population, Census Median Household Income and Median Sales Price data
# Cleaning process is based on the Excel formulas in the Excel sheet
for idx, row in keys_filtered.iterrows():

    key_row = row["key"]
    state_code = row["code"]
    state_code2 = row["code2"]
    state_blurb_prefix = row["blurb"]
    
    # --- Census Population ---
    # I find the row with the first column equal to "    Total population"
    # and then find the column named "<state_code>!!Estimate"
    pop_row = census_pop_df[census_pop_df.iloc[:, 0] == "    Total population"]
    pop_col_name = f"{state_code}!!Estimate"
    
    ## Extract the census population number for that particular "state_code" (State)
    ## If it doesn't exist in the data we leave it empty
    try:
        census_population = float(pop_row[pop_col_name].values[0])
    except Exception:
        census_population = np.nan

    # --- Census Median Household Income ---
    # I find the row where the first column equals "    Households"
    mhi_row = census_mhi_df[census_mhi_df.iloc[:, 0] == "    Households"]
    mhi_col_name = f"{state_code}!!Median income (dollars)!!Estimate"

    ## Extract the median household income for that particular "state_code" (State)
    ## If it doesn't exist in the data we leave it empty
    try:
        median_household_income = float(mhi_row[mhi_col_name].values[0])
    except Exception:
        median_household_income = np.nan

    # --- Redfin Median Sale Price ---
    # The data rows start at row index 2.
    # The first column (column 0) contains the state code.
    redfin_data = redfin_df.iloc[2:].copy()
    if state_code in redfin_data.iloc[:, 0].values:
        redfin_state_row = redfin_data[redfin_data.iloc[:, 0] == state_code]
    else:
        redfin_state_row = redfin_data[redfin_data.iloc[:, 0] == state_code2]

    try:
        raw_sale_price = str(redfin_state_row.iloc[0, last_valid_col])
        # Remove "$" and "K" (replace "K" with "000")
        cleaned_price = raw_sale_price.replace("$", "").replace("K", "000").replace(",", "")
        median_sale_price = float(cleaned_price)
    except Exception:
        median_sale_price = np.nan

    ## Dictionary to save all the data
    results.append({
        "key_row": key_row,
        "state_code": state_code,
        "census_population": census_population,
        "median_household_income": median_household_income,
        "median_sale_price": median_sale_price,
        "blurb_prefix": state_blurb_prefix
    })

# Create a dataset of all the results
df = pd.DataFrame(results)


# --- Ranking calculations ---
# For population, median household income, and median sale price,
# I use descending order (largest value gets rank 1).
df["population_rank"] = df["census_population"].rank(method="min", ascending=False)
df["mhi_rank"] = df["median_household_income"].rank(method="min", ascending=False)
df["sale_price_rank"] = df["median_sale_price"].rank(method="min", ascending=False)

# For house affordability ratio, calculate ratio and rank in ascending order (lowest ratio is best).
df["house_affordability_ratio"] = (df["median_sale_price"] / df["median_household_income"]).round(1)
df["affordability_rank"] = df["house_affordability_ratio"].rank(method="min", ascending=True)

# Convert numeric ranks to ordinal strings.
# I applied the ordinal function I created earlier (Line 6 - 14)
df["population_rank"] = df["population_rank"].apply(ordinal)
df["median_household_income_rank"] = df["mhi_rank"].apply(ordinal)
df["median_sale_price_rank"] = df["sale_price_rank"].apply(ordinal)
df["house_affordability_ratio_rank"] = df["affordability_rank"].apply(ordinal)


# --- Blurb construction ---
# Blurb was created based on the Excel function and the Rank calculated in 'Ranking Calculation'
df["population_blurb"] = df["blurb_prefix"] + " is " + df["population_rank"] + " in the nation in population among states, DC, and Puerto Rico."
df["median_household_income_blurb"] = df["blurb_prefix"] + " is " + df["median_household_income_rank"].replace("1st", "the highest") + " in the nation in median household income among states, DC, and Puerto Rico."
df["median_sale_price_blurb"] = (df["blurb_prefix"] +
                          " has the " +
                          df["median_sale_price_rank"].replace("1st", "single") +
                          " highest median sale price on homes in the nation among states, DC, and Puerto Rico, according to Redfin data from " +
                          redfin_date_str + ".")
df["house_affordability_ratio_blurb"] = (df["blurb_prefix"] +
                             " has the " +
                             df["house_affordability_ratio_rank"].replace("1st", "single") +
                             " lowest house affordability ratio in the nation among states, DC, and Puerto Rico, according to Redfin data from " +
                             redfin_date_str + ".")

# --- Select and order output columns ---
# I re-ordered the ouput columns to fit the output data given as an example
output_cols = ["key_row", "census_population", "population_rank", "population_blurb",
               "median_household_income", "median_household_income_rank", "median_household_income_blurb",
               "median_sale_price", "median_sale_price_rank", "median_sale_price_blurb",
               "house_affordability_ratio", "house_affordability_ratio_rank", "house_affordability_ratio_blurb"]

output_df = df[output_cols]


# --- Re-format the Numeric columns ---
# Format census_population as 000,000,000
output_df["census_population"] = output_df["census_population"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")

# Format median_household_income & median_sale_price as USD
output_df["median_household_income"] = output_df["median_household_income"].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "")
output_df["median_sale_price"] = output_df["median_sale_price"].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "")

# Final Output
## Excel Output -- This retains all number formats like $, etc.
output_df.to_excel("output.xlsx", index=False)

## CSV Output -- This doesn't retains number formats.
output_df.to_csv("output.csv", index=False)


# --- Basic Analysis ---

# Calculate correlations among key numeric columns
# - Household income and sale price have a strong positive correlation (~0.74).
# - Sale price and affordability ratio are very highly correlated (~0.89), meaning higher prices usually reduce affordability.
# - Population shows weak correlations with both income and sale price, suggesting it's not a strong driver for these values.

numeric_cols = ['census_population', 'median_household_income', 'median_sale_price', 'house_affordability_ratio']
corr_matrix = df[numeric_cols].corr()
print("Correlation Matrix:")
print(corr_matrix)


# --- Visualizations ---
# 1. Correlation heatmap
# Replace underscores with spaces for axis labels
labels = [label.replace('_', ' ').title() for label in corr_matrix.columns]

plt.figure(figsize=(16, 9))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    cbar_kws={'label': 'Correlation Coefficient'},
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Correlation Matrix of Demographic and Housing Variables", fontsize=16)
plt.xlabel("Variables", fontsize=14)
plt.ylabel("Variables", fontsize=14)
plt.xticks(rotation=25, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
# plt.savefig("Correlation Matrix Figure.png", dpi=300)
plt.show()


# 2. Scatter plot: Median Household Income vs. Median Sale Price
plt.figure(figsize=(16, 9))
ax = sns.scatterplot(data=df, x="median_household_income", y="median_sale_price", palette="tab10")

if ax.get_legend():
    ax.get_legend().remove()

# Annotate the top state based on the highest median sale price
largest = df.nlargest(1, 'median_sale_price')
for _, row in largest.iterrows():
    ax.annotate(
        row['key_row'].title(), 
        (row['median_household_income'], row['median_sale_price']),
        textcoords="offset points", 
        xytext=(-40, 10), 
        fontsize=12, 
        color='lightblue', 
        weight='bold'
    )

# Exclude rows with NA values in either median_household_income or median_sale_price (Puerto Rico)
df_clean = df.dropna(subset=['median_household_income', 'median_sale_price'])

# Annotate the state with the lowest median household income (after filtering out NA rows)
lowest = df_clean.nsmallest(1, 'median_household_income')
for _, row in lowest.iterrows():
    ax.annotate(
        row['key_row'].title(), 
        (row['median_household_income'], row['median_sale_price']),
        textcoords="offset points", 
        xytext=(-40, -15), 
        fontsize=12, 
        color='red', 
        weight='bold'
    )

plt.title("Median Sale Price vs. Median Household Income", fontsize=16)
plt.xlabel("Median Household Income", fontsize=14)
plt.ylabel("Median Sale Price", fontsize=14)

# Set the axis tick formatter to display values in USD with K and M suffixes
ax.xaxis.set_major_formatter(ticker.FuncFormatter(usd_formatter))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(usd_formatter))

plt.tight_layout()
# plt.savefig("Median Household Income vs. Median Sale Price.png", dpi=300)
plt.show()


# 3. Bar chart: Population by State
## Standardizing the state names
df_clean["state_label"] = df_clean["key_row"].str.replace("_", " ").str.title()
df_clean.loc[df_clean["key_row"] == "washington_dc", "state_label"] = "Washington DC"
df_sorted = df_clean.sort_values("census_population")

plt.figure(figsize=(16, 10))
ax = sns.barplot(
    x="census_population",
    y="state_label",
    data=df_sorted,
    palette="viridis"
)
plt.title("Population by State", fontsize=18)
plt.xlabel("Population", fontsize=14)
plt.ylabel("State", fontsize=14)

# Format x-axis ticks in K/M (Standard format for easy understanding by the users)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: 
    f"{x/1_000_000:.1f}M" if x >= 1_000_000 else f"{x/1_000:.0f}K" if x >= 1_000 else f"{int(x)}"
))

# Annotate only the smallest and largest bars with arrows
n = len(df_sorted)
for i, row in enumerate(df_sorted.itertuples()):
    if i == 0 or i == n - 1:
        bar_end = row.census_population
        x_text = bar_end + 5e5  
        y_text = i

        label = (
            f"{row.census_population/1_000_000:.1f}M"
            if row.census_population >= 1_000_000
            else f"{row.census_population/1_000:.0f}K"
        )
        color = "darkblue" if i == 0 else "lightgreen"

        ax.annotate(
            label,
            xy=(bar_end, y_text),
            xytext=(x_text, y_text),
            color=color,
            fontstyle="italic",
            weight="bold",
            fontsize=12,
            va="center",
            ha="left",
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
                shrinkA=0,
                shrinkB=0
            )
        )
plt.tight_layout()
plt.savefig("Population by State.png", dpi=300)
plt.show()
