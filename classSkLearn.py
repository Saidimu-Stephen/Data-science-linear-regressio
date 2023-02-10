import pandas as pd
from sklearn.datasets import load_breast_cancer
breast_cancer=load_breast_cancer()
print(breast_cancer)

# wrangling the data

print('\n\n\n wrangling the breast cancer data')
cancer_df=pd.DataFrame(breast_cancer.data)
print(cancer_df)


print('\n\n\n making it simpler for understanding, by refering to the taget')
cancer_df=pd.DataFrame(breast_cancer.target)
print(cancer_df)




# core api