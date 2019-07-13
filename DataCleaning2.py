from datascience import * 
import pandas as pd 
import numpy as np 
import re 

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
        return re.sub(r'(,\w*\s*\w*)+', '', entry)
    assert (isinstance(table, Table)), "Input not a supported type."
    column = table.apply(replace, column_name)
    return table.append_column(column_name, column) 
    
    

def missing_proportion(table, column_name):
	"""Takes in a table and column name whose column pontentially has missing (NaN) values and returns 
	the proportion of missing values in that column, rounded to 3 decimal places. Supports pandas Series 
	and Datascience Tables.
	"""
	assert (isinstance(table, Table) or isinstance(table, pd.DataFrame)), "Input not a supported type."
	if isinstance(table, pd.DataFrame):
		assert (column_name in list(table.columns.values)), "Input a valid column name."
		total_missing = table[column_name].isna().sum()
		prop_missing = total_missing/len(table)
		return round(prop_missing, 2)
	else:
		encode_nans(table, column_name)
		assert (column_name in list(table.labels)), 'Input a valid column name.'
		prop_missing = table.where(column_name, pd.isnull).num_rows/table.num_rows
		return round(prop_missing, 2)



def drop_missing_rows(table, column_name = None):
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
