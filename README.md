# Structural Protein Sequences

### Overview of the libraries and functions we imported:

- NumPy (np): A library for numerical operations in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.

- Pandas (pd): A powerful data manipulation and analysis library. It provides data structures like DataFrame for efficient data handling.

- Seaborn (sns): A statistical data visualization library based on Matplotlib. It provides an interface for creating attractive and informative statistical graphics.

- Matplotlib (plt): A comprehensive library for creating static, animated, and interactive visualizations in Python.

- Missingno (msno): A library for visualizing missing data in a dataset. It helps to understand the distribution of missing values in a dataset.

- Warnings: The warnings module is used to filter out and ignore warnings during execution.

- Scikit-learn (sklearn): A machine learning library that includes various tools for classification, regression, clustering, and more.
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization
import missingno as msno # visualizatin for missing values

import warnings
warnings.filterwarnings("ignore") # ignore warnings

from sklearn.model_selection import train_test_split # train and test split

from sklearn.impute import KNNImputer # filling missing data with KNN method

from sklearn.preprocessing import LabelEncoder # filling missing categorical values with label encoder method
import category_encoders as ce

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
```python
# load file 
df = pd.read_csv("../../..") #path of the dataset
```

## Variable Description
```
structureId: identity of the structure
classification: classification type
experimentalTechnique: technique of experiment
macromoleculeType: type of macromolecule
residueCount: number of residue
resolution: amount of resolution
structureMolecularWeight: molecular weight
crystallizationMethod: method of crystallization
crystallizationTempK: crystallization temperature in Kelvin
densityMatthews: crystalline density
densityPercentSol: resolution ratio by density
pdbxDetails: detail about row
phValue: PH value
publicationYear: published year
structureId: identity of the structure
classification: classification type
experimentalTechnique: technique of experiment
macromoleculeType: type of macromolecule
residueCount: number of residue
resolution: amount of resolution
structureMolecularWeight: molecular weight
crystallizationMethod: method of crystallization
crystallizationTempK: crystallization temperature in Kelvin
densityMatthews: crystalline density
densityPercentSol: resolution ratio by density
pdbxDetails: detail about row
phValue: PH value
publicationYear: published yearstructureId: identity of the structure
classification: classification type
experimentalTechnique: technique of experiment
macromoleculeType: type of macromolecule
residueCount: number of residue
resolution: amount of resolution
structureMolecularWeight: molecular weight
crystallizationMethod: method of crystallization
crystallizationTempK: crystallization temperature in Kelvin
densityMatthews: crystalline density
densityPercentSol: resolution ratio by density
pdbxDetails: detail about row
phValue: PH value
publicationYear: published year
```
```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 141401 entries, 0 to 141400
Data columns (total 14 columns):
 #   Column                    Non-Null Count   Dtype  
---  ------                    --------------   -----  
 0   structureId               141401 non-null  object 
 1   classification            141399 non-null  object 
 2   experimentalTechnique     141401 non-null  object 
 3   macromoleculeType         137636 non-null  object 
 4   residueCount              141401 non-null  int64  
 5   resolution                128589 non-null  float64
 6   structureMolecularWeight  141401 non-null  float64
 7   crystallizationMethod     96242 non-null   object 
 8   crystallizationTempK      97039 non-null   float64
 9   densityMatthews           124724 non-null  float64
 10  densityPercentSol         124749 non-null  float64
 11  pdbxDetails               118534 non-null  object 
 12  phValue                   105110 non-null  float64
 13  publicationYear           117602 non-null  float64
dtypes: float64(7), int64(1), object(6)
memory usage: 15.1+ MB
```
Removing unnecessary column.(pdbxDetails)
```python
df.drop("pdbxDetails", axis=1, inplace = True)
```

A function named missing_value_table to create a table visualizing missing values in a DataFrame (df). The function calculates the count and percentage of missing values for each column and presents the information in a styled Pandas DataFrame with a color gradient.
```
# Missing Value Table
def missing_value_table(df):
    missing_value = df.isna().sum().sort_values(ascending=False)
    missing_value_percent = 100 * df.isna().sum()//len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})
    cm = sns.light_palette("lightgreen", as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return
  
missing_value_table(df)
```
This function generates a styled DataFrame with missing values highlighted based on the percentage of missing data. The color gradient is applied to make it visually informative.

<img width="196" alt="image" src="https://github.com/IDrDomino/Structural-Protein-Sequences/assets/154571800/5b8e0429-87e1-47fd-ac58-5817fb149bfa">

Now, the data visualization part, 
```python
sns.pairplot(df)
sns.set(style="ticks", color_codes=True)
```
![__results___26_0](https://github.com/IDrDomino/Structural-Protein-Sequences/assets/154571800/acc7ef58-7721-438e-a9ef-b51b528882bc)

- pairplot to compare each column

Now, the correlation matrix for the columns in your DataFrame (df) and then visualizing the correlation using a heatmap with the Seaborn library.

```python
corr = df.corr()
plt.figure(figsize=(12,5))
sns.heatmap(corr, annot=True)
```

![__results___29_1](https://github.com/IDrDomino/Structural-Protein-Sequences/assets/154571800/f9da61e4-1cf6-43a9-bf42-92af0ddc2dae)

Analyzing the correlation between numerical values reveals that light colors indicate higher correlations. Notably, there is a strong correlation of 0.84 between densityMatthews and densityPercentSol. Additionally, a noteworthy correlation worth considering is 0.55 between structureMolecularWeight and residueCount. Light colors on the correlation matrix signify stronger associations between the mentioned pairs of numerical values.

## MacromoleculeType

Representing the distribution of values in the 'macromoleculeType' column, focusing on the top 5 types. The explosion effect is achieved with the explode parameter.
```python
plt.figure(figsize=(20,18))
ex = df.macromoleculeType.value_counts(ascending=False)[:5]
figureObject, axesObject = plt.subplots() 
explode = (0.2, 0.5, 0.5, 0.5, 0.5)
plt.title("Macro Molecule Type",color = 'darkblue',fontsize=15)

axesObject.pie(ex.values,
               labels   = ex.index,
               shadow   = True,                       
               explode  = explode,
               autopct  = '%.1f%%',
               wedgeprops = { 'linewidth' : 3,'edgecolor' : "orange" })                              
             
axesObject.axis('equal') 

plt.show()
```

![__results___36_1](https://github.com/IDrDomino/Structural-Protein-Sequences/assets/154571800/78db3900-7cfa-4096-9d29-733732d685bc)

- According to the chart, proteins are the most prevalent type of molecule.

- Filling train macromoleculeType with mode.

```python
x_train_copy['macromoleculeType'].fillna(x_train_copy['macromoleculeType'].mode()[0], inplace=True) # fill missing data with mode
```

- Mean and Std Method

Handling missing values in the 'resolution' column of your DataFrame (x_train) by imputing them with random values generated from a uniform distribution within the range defined by the standard deviation (x_train_resol_std) and mean (x_train_resol_mean) of the existing non-null values.

```python
x_train_resol_std, x_train_resol_mean = x_train_copy.resolution.std(), x_train_copy.resolution.mean() # mean and standard deviation
random = np.random.uniform(x_train_resol_std, x_train_resol_mean, 113120) # 113120 numbers of rows 
x_train.resolution = x_train.resolution.mask(x_train.resolution.isnull(), random)
```

Applying a similar strategy to handle missing values in the 'resolution' column for your test dataset (x_test). You are calculating the mean and standard deviation of the 'resolution' column in the test dataset, generating random numbers from a uniform distribution within the specified range, and then replacing missing values with these random values.

```python
x_test_resol_std, x_test_resol_mean = x_test_copy.resolution.std(), x_test_copy.resolution.mean() # mean and standard deviation
random = np.random.uniform(x_test_resol_std, x_test_resol_mean, 28281) # 28281 numbers of rows 
x_test.resolution = x_test.resolution.mask(x_test.resolution.isnull(), random)
```




