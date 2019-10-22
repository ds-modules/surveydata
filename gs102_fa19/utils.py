from wordcloud import *
from datascience import *
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set_style("whitegrid")

"""Summer 2019 Data Science Education Team
Helper Functions for Data Cleaning Notebook
Class: GLOBAL 150Q
"""


def encode_nans(table, column_name):
    """Takes in a Table and column name and converts all the string NaN entries to standard None types
    which are easier for detection by null checker methods."""
    def replace(entry):
        if entry == 'nan' or pd.isnull(entry):
            return None
        else:
            return entry
    assert (isinstance(table, Table)), "Input not a supported type."
    column = table.apply(replace, column_name)
    return table.append_column(column_name, column)



def encode_nans_table(table):
    """Takes in a Table and converts all the string NaN entries to standard None types
        which are easier for detection by null checker methods."""
    col_names = table.labels
    for col in col_names :
        table = encode_nans(table, col)

    return table



def to_numerical(table, column_name):
    """Takes in a Table and column name and converts all the numbera coded as stringa to integer."""
    def replace(entry):
        return float(entry)
    assert (isinstance(table, Table)), "Input not a supported type."
    column = table.apply(replace, column_name)
    return table.append_column(column_name, column)


def get_first_selection(table, column_name):
    """Takes in a Table and column name, get the first selection of participants for a question."""
    def replace(entry):
        if pd.isnull(entry):
            return None
        else:
            return re.sub(r',.*', '', entry)
    assert (isinstance(table, Table)), "Input not a supported type."
    column = table.apply(replace, column_name)
    return table.append_column(column_name, column)


def get_mixed_category(table, column_name, string):
    """Takes in a Table and column name, if the parcitipate choose more than one answer, label it as 'Mixed String' category."""
    def replace(entry):
        if pd.isnull(entry):
            return None
        elif ',' in entry:
            return 'Mixed ' + string
        else:
            return entry
    assert (isinstance(table, Table)), "Input not a supported type."
    column = table.apply(replace, column_name)
    return table.append_column(column_name, column)


def convert_degree_to_num(table, column_name, levels):
    """Takes in a Table, a column name, and a list of strings of variable levels.
       Modifies the input table, replace the column_name with a new column of numerical values representing degree."""
    encode_nans(table, column_name)
    input_level_list = pd.Series(levels).unique()
    level_list = pd.Series(table.column(column_name)).unique()
    assert (len(levels) == len(level_list) | len(input_level_list) == len(level_list)
            ), "Input list have differnt number of levels, please double check your input levels."
    assert (sum([entry not in level_list for entry in levels]) ==
            0), "Input list has different items, please check your spelling and level name."

    for i in range(len(levels)):
        table.column(column_name)[table.column(column_name) == levels[i]] = i

def convert_num_to_text(table, column_name, values):
    if table.column(column_name).dtype == np.dtype('<U16'):
        return
    encode_nans(table, column_name)
    input_level_list = pd.Series(values).unique()
    level_list = pd.Series(table.column(column_name)).dropna().unique()
    assert (len(values) == len(level_list) | len(input_level_list) == len(level_list)
            ), "Input list have differnt number of levels, please double check your input levels."
    new_values = [''] * len(table.column(column_name))
    for i in range(len(table.column(column_name))):
        curr_value = int(table.column(column_name)[i])
        new_values[i] = values[curr_value - 1]
        
    table[column_name] = new_values

def missing_proportion(table, column_name):
    """Takes in a table and column name whose column pontentially has missing (NaN) values and returns
    the proportion of missing values in that column, rounded to 3 decimal places. Supports pandas Series
    and Datascience Tables.
    """
    assert (isinstance(table, Table) or isinstance(
        table, pd.DataFrame)), "Input not a supported type."
    if isinstance(table, pd.DataFrame):
        assert (column_name in list(table.columns.values)
                ), "Input a valid column name."
        total_missing = table[column_name].isna().sum()
        prop_missing = total_missing/len(table)
        return round(prop_missing, 2)
    else:
        encode_nans(table, column_name)
        assert (column_name in list(table.labels)
                ), 'Input a valid column name.'
        prop_missing = table.where(
            column_name, pd.isnull).num_rows/table.num_rows
        return round(prop_missing, 2)


def drop_nonserious_rows(table, column_name):
    """Takes in a table and column name. Returns a new table without rows that have missing values for all the columns that start from the given column_name.
    """
    # encode table's nan as None
    for column in table.labels:
        encode_nans(table, column)
    full_df = table.to_df()
    start_idx = table.column_index(column_name)
    tbl = table.select(range(start_idx, table.num_columns))
    df = tbl.to_df()
    na_df = df.notna()
    full_df = full_df[np.array(na_df.apply(np.sum, axis=1) != 0)]
    return Table.from_df(full_df)


def drop_missing_rows(table, column_name=None):
    """Takes in Datascience Table with pontentially with missing (NaN) values and drops the rows which
    contain missing values with respect to a particular column. Returns the resulting table.
    """
    assert isinstance(table, Table), "Input not a supported type."
    assert (column_name in table.labels), "Input is invalid. Enter a valid column name."
    encode_nans(table, column_name)
    omitted_indices = []
    for index in range(len(table.column(column_name))):
        if pd.isnull(table.column(column_name).item(index)):
            omitted_indices.append(index)
    new_table = Table(np.asarray(table.labels))
    new_table = new_table.with_row(list(np.zeros(table.num_columns)))
    counter = 0
    for row in table.rows:
        if counter in omitted_indices:
            counter += 1
            continue
        else:
            new_table = new_table.with_row(row)
            counter += 1
    new_table = new_table.exclude(0)
    return new_table


def extract_first_major(major):
    if '/' in major:
        return major.split('/')[0]
    else:
        return major


def fix_major_formatting(data, column_name):
    return data.with_column(column_name, data.apply(
        extract_first_major, column_name))


def counts_to_proportions(pivot):
    first_name = pivot.labels[0]
    first_values = pivot.column(first_name)
    pivot = pivot.drop(first_name)
    pivot_df = pivot.to_df()
    sums = np.array([])
    for col in pivot.labels:
        sums = np.append(sums, np.sum(pivot.column(col)))
    table2 = pivot_df.div(sums)
    new_pivot = Table.from_df(table2)
    new_pivot.append_column(first_name, first_values)
    new_pivot.move_to_start(first_name)
    return new_pivot


def plot_bar_chart(pivot, columns, title, category):
    first_name = pivot.labels[0]
    first_values = pivot.column(first_name)
    pivot = pivot.drop(first_name)
    filtered = pivot.select(columns)
    filtered = filtered.with_column(first_name, first_values)
    filtered.move_to_start(first_name)
    df = filtered.to_df()
    df2 = pd.melt(df, id_vars=[first_name])
    df2 = df2.rename(columns = {"variable":category})
    sns.set(font_scale=1.5)
    sorted_list = [x for x in np.unique(df2[category])]
    first_type  = type(sorted_list[0])
    sorted_list = sorted(sorted_list, key = first_type)
    sns.factorplot(x=first_name, y='value', hue=category,
                   data=df2, kind='bar', size=9, aspect=1.5,
                   hue_order = [str(y) for y in sorted_list])
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Proportion')


def create_wordcloud(table, column):
    all_words = ""
    for word in drop_missing_rows(table, column).column(column):
        if ',' in word:
            for part in word.split(','):
                all_words += part.lower() + " "
        else:
            all_words += word.lower() + " "
    wordcloud = WordCloud(collocations = False).generate(all_words)
    plt.title('Wordcloud for {}'.format(column))
    plt.axis('off')
    plt.imshow(wordcloud)

def load_dataset():
    party_by_gender = Table(np.array(['Party', 'Gender']))
    male_dem = list(np.array(['Democrat', 'Male']))
    male_rep = list(np.array(['Republican', 'Male']))
    fem_dem = list(np.array(['Democrat', 'Female']))
    fem_rep = list(np.array(['Republican', 'Female']))
    for _ in range(20):
        party_by_gender = party_by_gender.with_row(male_dem)
        party_by_gender = party_by_gender.with_row(fem_rep)
    for _ in range(30):
        party_by_gender = party_by_gender.with_row(male_rep)
        party_by_gender = party_by_gender.with_row(fem_dem)
    return party_by_gender

def add_row_totals(table):
    row_totals = np.array([])
    for i in np.arange(table.num_rows):
        all_counts = np.array(table.row(i)[1:])
        row_sums = sum(all_counts)
        row_totals = np.append(row_totals, row_sums)
    return table.with_column('Row Total', row_totals)

def add_column_totals(table):
    data = table.drop(0)
    col_sums = []
    col_sums.append('Column Total')
    for label in data.labels:
        total = np.sum(data.column(label))
        col_sums.append(total)
    return table.with_row(col_sums)

def chi_expected_values(contingency):
    """Calculates expected values of contingency table, given that the table
    is in proper format, with Row Total column and Column Total row."""
    #Deprecated
    total_responses = contingency.column(contingency.num_columns - 1).item(-1)
    expected_values = []
    for col_index in range(1, contingency.num_columns - 1):
        for row_index in range(0, contingency.num_rows - 1):
            column_total = contingency.column(col_index).item(-1)
            row_total = contingency.row(row_index).item(-1)
            expected_values.append((column_total*row_total)/total_responses)
    return np.array(expected_values)

def prepare_pivot_values(contingency):
    """Returns actual values of contingency table, given that the table
    is in proper format, with Row Total column and Column Total row"""
    actual_values = []
    for col_index in range(1, contingency.num_columns - 1):
        col_data = []
        for row_index in range(0, contingency.num_rows - 1):
            col_data.append(contingency[col_index][row_index])
        actual_values.append(col_data)
    return np.array(actual_values)

def chisquaretest(contingency):
    values = prepare_pivot_values(contingency)
    results = chi2_contingency(values, correction = False)
    
    output = {'expected': results[3],
             'chi2 statistic': results[0],
             'p-value': results[1]}
    return output