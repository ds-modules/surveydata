from datascience import * 
import pandas as pd 
import numpy as np 

"""Summer 2019 Data Science Education Team 
Helper Functions for Data Cleaning Notebook 
Class: GLOBAL 150Q
"""

def missing_proportion(col_data):
	"""Takes in column data pontentially with missing (NaN) values and returns the proportion
	of missing values rounded to 3 decimal places. Supports pandas Series and Datascience Table.

	>>> series = pd.Series([1, np.nan, 3])
	>>> missing_proportion(series)
	0.333

	>>> table_column = Table.from_columns_dict({'col_1':[np.nan, 0 ,1]})
	>>> missing_proportion(col_data)
	0.333
	"""
	assert (isinstance(col_data, Table) or isinstance(col_data, pd.Series)), "Input not a supported type."
	if isinstance(col_data, Table):
		assert (col_data.num_columns == 1), "Too many columns. Input a table with only one column."
	if isinstance(col_data, pd.Series):
		total_missing = col_data.isna().sum()
		prop_missing = total_missing/len(col_data)
		return round(prop_missing, 3)
	else:
		prop_missing = col_data.where(0, np.isnan).num_rows/col_data.num_rows
		return round(prop_missing, 3)

def drop_missing(table, columns = False):
	"""Takes in table with pontentially with missing (NaN) values and drops the rows or columns which
	contain missing values. Returns the resulting table. Supports pandas DataFrames and datascience 
	Tables. By default, rows with missing values will be dropped unless columns = True is specified
	as an argument. 

	>>> table = pd.DataFrame({"first":[1, np.nan, 3], "second":[1, 2, 3]})
	>>> drop_missing(table)
		first	second	
	0     1        1
	1     3        3
		
	>>> table = pd.DataFrame({"first":[1, np.nan, 3], "second":[1, 2, 3]})
	>>> drop_missing(table, columns = True)
		second
	0	   1
	1	   2
	2	   3

	>>> table = Table.from_columns_dict({"first":[1, np.nan, 3], "second":[1, 2, 3]})
	>>> drop_missing(table)
	first	second
	  1        1
	  3        3

	>>> table = Table.from_columns_dict({"first":[1, np.nan, 3], "second":[1, 2, 3]})
	>>> drop_missing(table, columns  = True)
	second
	   1
	   2
	   3
	"""
	assert (isinstance(table, Table) or isinstance(table, pd.DataFrame)), "Input not a supported type."
	assert (isinstance(columns, bool)), "Input is invalid. Enter either True or False."
	if isinstance(table, Table):
		if columns == False:
			new_table = Table(np.asarray(table.labels))
			new_table = new_table.with_row(list(np.zeros(table.num_columns)))
			for row in table.rows:
				if True in np.isnan(np.array(row)):
					continue
				else:
					new_table = new_table.with_row(row)
			new_table = new_table.exclude(0)
			return new_table
		else:
			new_table = Table()
			a = [label for label in table.labels]
			b = [col for col in table.columns]
			for label, col in list(zip(a, b)):
				if True in np.isnan(np.array(col)):
					continue
				else:
					new_table = new_table.with_column(label, col)
			return new_table
	else:
		if columns == False:
			return table.dropna()
		else:
			return table.dropna(axis = 1)