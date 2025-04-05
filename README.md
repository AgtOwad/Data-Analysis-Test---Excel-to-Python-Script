# Trial Assignment - Data Analyst

## Overview
This Python script `DataProcessing.py` replicates the functionality of the `OUTPUT` tab in the provided Google Sheet by:

- Loading and cleaning four source datasets (Census population, median household income, Redfin median sale price, and keys).
- Applying transformations and ranking logic to match spreadsheet formulas.
- Generating descriptive blurbs for each state.
- Producing `output.csv` and `output.xlsx` with formatted results.
- Performing basic data analysis and visualizing insights.

## Data Sources
- **KEYS.csv**: Mapping of state keys, codes, and blurb prefixes.  
- **CENSUS_POPULATION_STATE.xlsx**: Total population estimates by state.  
- **CENSUS_MHI_STATE.xlsx**: Median household income estimates by state.  
- **REDFIN_MEDIAN_SALE_PRICE.csv**: Monthly median home sale prices by state.  

## Key Steps
1. **Data Loading**  
   Read CSV/XLSX files using Pandas, preserving formatting where necessary.  
2. **Data Cleaning & Extraction**  
   - Filter keys for relevant state entries.  
   - Locate rows and columns in Census and Redfin tables based on codes.  
   - Convert raw strings (e.g., `$350K`) to numeric values.  
3. **Ranking & Formatting**  
   - Compute descending ranks for population, income, and sale price.  
   - Calculate house affordability ratio and its ascending rank.  
   - Convert numeric ranks to ordinal strings (e.g., `1st`, `2nd`).  
4. **Blurb Generation**  
   Construct descriptive sentences per state using rank and prefix.  
5. **Output**  
   - Export formatted results to `output.xlsx` (with number formats) and `output.csv`.  
6. **Analysis & Visualizations**  
   - Correlation heatmap of key numeric variables.  
   - Scatter plot of median household income vs. median sale price, highlighting extremes.  
   - Bar chart of state populations, annotated for smallest and largest.  

## Visualizations
- **Correlation Matrix**: Shows relationships among population, income, sale price, and affordability.  
- **Income vs. Sale Price Scatter**: Identifies the states with highest sale price and lowest income.  
- **Population Bar Chart**: Ranks states by population with annotations for top and bottom.  

## Requirements
- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- seaborn  

## Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn
