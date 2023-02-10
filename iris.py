# there are 4 preprocessing steps
# 1. data extraction
# 2. understanding the data
# 3. data transformation
# 4. data integration

import pandas as pd

data = pd.read_csv('/home/saidimu/Desktop/Machin learning/iris.csv')  # data  extraction
print(data.columns)
print(data.head())
print(data.tail())
print(data.sample(10))
print(data.shape)
print(data[["SepalLengthCm"]])
print(data[["Species"]])

print(data)
print(data['Species'].value_counts())

# measure of central tendency
# count
# median
# mean

print(data.count())

print("minimum ")
print(data.min())  # smallest value

print(data.max())  # maximu value
print(data['SepalLengthCm'].mean())

print('\n mode')
print(data.mode())  # most appearing value

print('\n\n measure of dispersion \n standard deviating \n Quantiles \n')
# print(data.std())
print(data.dtypes)
print(data.describe())

print(data['SepalLengthCm'].mean())

# renaming the columns
new_columns = {'Id': 'id',
               'SepalLengthCm': 'S_Length',
               'SepalWidthCm': 'S_Width',
               'PetalLengthCm': 'P_Length',
               'PetalWidthCm': 'P_Width'}
data.rename(columns=new_columns, inplace=True)
print(data.columns)

# cleaning data
# problems that can be encountered in data set

# a. missing values
# 1. fill the missing values with the  mean value/ modal value(a value that is frequently appearing in that column)
# 2. Using intuition.
# 3. Omit the columns with the missing values

# b. duplicate values
# we remove them

# c. outliers
# drop them, replace them

# d. misspelled names


# DATA TRANSfORMATION
# Labeling
# Change none numeric data to numbers

# data splitting
# 1. training data set
# 2. testing the data set


# normalizing of data
# rescaling


# standardization
# rescale the data to have a normal distribution
