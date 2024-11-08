import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
cropdf = pd.read_csv('3MTTCropRec.csv')
cropdf.isnull().sum()
N              0
P              0
K              0
temperature    0
humidity       0
ph             0
rainfall       0
Crops          0
dtype: int64
label_encoder = LabelEncoder()
cropdf['crop_encoded'] = label_encoder.fit_transform(cropdf['Crops'])
print(cropdf)
        N   P   K  temperature   humidity        ph     rainfall    Crops  \
0      92  55  55    26.920279  75.367074  5.342474  1288.647218  Cassava   
1     117  46  60    23.853979  65.879359  5.496187  1080.899834  Cassava   
2     105  48  55    24.510217  72.973813  5.520667  1333.156601  Cassava   
3     102  52  66    23.430753  78.717684  5.527641  1288.268276  Cassava   
4      86  44  60    24.520497  61.819748  5.563377  1242.304099  Cassava   
...   ...  ..  ..          ...        ...       ...          ...      ...   
1995  107  34  32    26.774637  66.413269  6.780064   177.774507   coffee   
1996   99  15  27    27.417112  56.636362  6.086922   127.924610   coffee   
1997  118  33  30    24.131797  67.225123  6.362608   173.322839   coffee   
1998  117  32  34    26.272418  52.127394  6.758793   127.175293   coffee   
1999  104  18  30    23.603016  60.396475  6.779833   140.937041   coffee   

      crop_encoded  
0                1  
1                1  
2                1  
3                1  
4                1  
...            ...  
1995             9  
1996             9  
1997             9  
1998             9  
1999             9  

[2000 rows x 9 columns]
cropdf
N	P	K	temperature	humidity	ph	rainfall	Crops	crop_encoded
0	92	55	55	26.920279	75.367074	5.342474	1288.647218	Cassava	1
1	117	46	60	23.853979	65.879359	5.496187	1080.899834	Cassava	1
2	105	48	55	24.510217	72.973813	5.520667	1333.156601	Cassava	1
3	102	52	66	23.430753	78.717684	5.527641	1288.268276	Cassava	1
4	86	44	60	24.520497	61.819748	5.563377	1242.304099	Cassava	1
...	...	...	...	...	...	...	...	...	...
1995	107	34	32	26.774637	66.413269	6.780064	177.774507	coffee	9
1996	99	15	27	27.417112	56.636362	6.086922	127.924610	coffee	9
1997	118	33	30	24.131797	67.225123	6.362608	173.322839	coffee	9
1998	117	32	34	26.272418	52.127394	6.758793	127.175293	coffee	9
1999	104	18	30	23.603016	60.396475	6.779833	140.937041	coffee	9
2000 rows × 9 columns

cropdf
N	P	K	temperature	humidity	ph	rainfall	Crops	crop_encoded
0	92	55	55	26.920279	75.367074	5.342474	1288.647218	Cassava	1
1	117	46	60	23.853979	65.879359	5.496187	1080.899834	Cassava	1
2	105	48	55	24.510217	72.973813	5.520667	1333.156601	Cassava	1
3	102	52	66	23.430753	78.717684	5.527641	1288.268276	Cassava	1
4	86	44	60	24.520497	61.819748	5.563377	1242.304099	Cassava	1
...	...	...	...	...	...	...	...	...	...
1995	107	34	32	26.774637	66.413269	6.780064	177.774507	coffee	9
1996	99	15	27	27.417112	56.636362	6.086922	127.924610	coffee	9
1997	118	33	30	24.131797	67.225123	6.362608	173.322839	coffee	9
1998	117	32	34	26.272418	52.127394	6.758793	127.175293	coffee	9
1999	104	18	30	23.603016	60.396475	6.779833	140.937041	coffee	9
2000 rows × 9 columns

from sklearn.model_selection import train_test_split
x = cropdf[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
x
N	P	K	temperature	humidity	ph	rainfall
0	92	55	55	26.920279	75.367074	5.342474	1288.647218
1	117	46	60	23.853979	65.879359	5.496187	1080.899834
2	105	48	55	24.510217	72.973813	5.520667	1333.156601
3	102	52	66	23.430753	78.717684	5.527641	1288.268276
4	86	44	60	24.520497	61.819748	5.563377	1242.304099
...	...	...	...	...	...	...	...
1995	107	34	32	26.774637	66.413269	6.780064	177.774507
1996	99	15	27	27.417112	56.636362	6.086922	127.924610
1997	118	33	30	24.131797	67.225123	6.362608	173.322839
1998	117	32	34	26.272418	52.127394	6.758793	127.175293
1999	104	18	30	23.603016	60.396475	6.779833	140.937041
2000 rows × 7 columns

y = cropdf['crop_encoded']
y.shape
(2000,)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
(1600, 7) (400, 7)
(1600,) (400,)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

  RandomForestClassifier?i
RandomForestClassifier()
y_pred = rfc.predict(X_test)
y_pred
array([11,  3, 14, 17, 19, 19, 17, 10,  1, 14,  1, 19, 13,  5,  3,  4, 18,
        6,  8, 11,  2,  8,  9,  7,  9,  1, 13, 17, 18,  1,  2, 17,  5, 17,
        7,  0,  8, 19,  7,  4,  6,  9,  4,  5, 16, 12,  5, 12,  1,  8, 16,
        4, 15,  4, 17, 10, 19,  8, 13,  3, 15,  2,  8,  7,  9, 11, 17, 16,
        6, 15, 15,  3, 14,  7, 18,  6,  8, 18,  3, 11,  1, 12, 15, 16,  8,
       15, 10,  0,  4,  8, 16, 10,  4,  4, 11, 11,  2, 19, 16,  9, 17, 17,
        0, 14,  4, 12,  2,  4,  2,  4, 18, 15,  7, 10, 18,  3, 13, 17,  3,
        8, 11,  7, 13,  0,  3, 16,  0, 12, 13,  3, 11, 11, 18,  5, 12, 19,
        9,  9,  8, 16, 14,  3,  8, 12, 16,  5,  2, 19,  4,  6,  8, 17,  2,
       13,  3,  6,  5,  8,  5,  8,  2, 12,  6, 13,  9, 15, 10, 19,  4, 10,
       13,  2,  9, 11,  0, 14, 18, 14, 12,  8, 19, 18, 11, 13, 12, 18,  9,
        8, 13, 19,  2,  4, 14,  4,  1,  0, 13, 12,  2, 16, 11,  5,  3, 18,
        0,  4, 15, 14,  7,  1, 10,  1,  4,  6, 18, 11, 13,  3,  0, 19, 17,
        6,  5, 13, 11,  9,  0,  6,  8,  3,  9,  4,  8, 15, 10,  3,  7, 12,
       14,  5, 18, 16,  5, 17, 14,  6, 12,  9, 12, 14,  7,  3, 13,  0, 15,
        7,  5,  2, 11, 19,  1,  9,  1, 16,  7, 16,  3,  6,  4,  7, 14,  9,
        0,  8,  9,  9,  2,  0, 15,  4,  9, 15,  2, 14, 14,  1,  4, 10,  1,
        6, 19, 16, 15,  8,  5,  3,  1, 10, 11,  0,  5,  1,  9, 15, 18,  9,
       13, 14, 17,  1,  7,  8, 19, 17,  7,  5, 16, 16, 11, 16,  1,  8,  6,
        3, 17,  5,  9,  2,  8, 18,  0,  5, 16,  5,  3, 10,  3,  6, 13, 10,
       15,  3,  9, 12,  5,  8,  3,  9, 16, 17,  6,  0, 15,  8,  6,  4, 10,
       10, 14, 13,  0,  3, 18,  6,  2,  0, 10,  3, 13,  4, 16, 11,  1, 10,
       16, 16,  0, 10,  9, 15, 15, 12,  0,  8, 15, 18,  5, 15,  2, 16, 10,
        8,  0, 18,  5, 17, 19, 10,  4,  5])
y_test
1860    11
353      3
1333    14
905     17
1289    19
        ..
965     17
1284    19
1739    10
261      4
535      5
Name: crop_encoded, Length: 400, dtype: int32
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))
0.99
crop_names = label_encoder.inverse_transform(y_pred)
crop_names
array(['jute', 'Plantains', 'muskmelon', 'pigeonpeas', 'watermelon',
       'watermelon', 'pigeonpeas', 'cotton', 'Cassava', 'muskmelon',
       'Cassava', 'watermelon', 'mango', 'Tomatoes', 'Plantains',
       'Potatoes', 'rice', 'Yam', 'coconut', 'jute', 'Pepper', 'coconut',
       'coffee', 'banana', 'coffee', 'Cassava', 'mango', 'pigeonpeas',
       'rice', 'Cassava', 'Pepper', 'pigeonpeas', 'Tomatoes',
       'pigeonpeas', 'banana', 'Carrot', 'coconut', 'watermelon',
       'banana', 'Potatoes', 'Yam', 'coffee', 'Potatoes', 'Tomatoes',
       'papaya', 'maize', 'Tomatoes', 'maize', 'Cassava', 'coconut',
       'papaya', 'Potatoes', 'orange', 'Potatoes', 'pigeonpeas', 'cotton',
       'watermelon', 'coconut', 'mango', 'Plantains', 'orange', 'Pepper',
       'coconut', 'banana', 'coffee', 'jute', 'pigeonpeas', 'papaya',
       'Yam', 'orange', 'orange', 'Plantains', 'muskmelon', 'banana',
       'rice', 'Yam', 'coconut', 'rice', 'Plantains', 'jute', 'Cassava',
       'maize', 'orange', 'papaya', 'coconut', 'orange', 'cotton',
       'Carrot', 'Potatoes', 'coconut', 'papaya', 'cotton', 'Potatoes',
       'Potatoes', 'jute', 'jute', 'Pepper', 'watermelon', 'papaya',
       'coffee', 'pigeonpeas', 'pigeonpeas', 'Carrot', 'muskmelon',
       'Potatoes', 'maize', 'Pepper', 'Potatoes', 'Pepper', 'Potatoes',
       'rice', 'orange', 'banana', 'cotton', 'rice', 'Plantains', 'mango',
       'pigeonpeas', 'Plantains', 'coconut', 'jute', 'banana', 'mango',
       'Carrot', 'Plantains', 'papaya', 'Carrot', 'maize', 'mango',
       'Plantains', 'jute', 'jute', 'rice', 'Tomatoes', 'maize',
       'watermelon', 'coffee', 'coffee', 'coconut', 'papaya', 'muskmelon',
       'Plantains', 'coconut', 'maize', 'papaya', 'Tomatoes', 'Pepper',
       'watermelon', 'Potatoes', 'Yam', 'coconut', 'pigeonpeas', 'Pepper',
       'mango', 'Plantains', 'Yam', 'Tomatoes', 'coconut', 'Tomatoes',
       'coconut', 'Pepper', 'maize', 'Yam', 'mango', 'coffee', 'orange',
       'cotton', 'watermelon', 'Potatoes', 'cotton', 'mango', 'Pepper',
       'coffee', 'jute', 'Carrot', 'muskmelon', 'rice', 'muskmelon',
       'maize', 'coconut', 'watermelon', 'rice', 'jute', 'mango', 'maize',
       'rice', 'coffee', 'coconut', 'mango', 'watermelon', 'Pepper',
       'Potatoes', 'muskmelon', 'Potatoes', 'Cassava', 'Carrot', 'mango',
       'maize', 'Pepper', 'papaya', 'jute', 'Tomatoes', 'Plantains',
       'rice', 'Carrot', 'Potatoes', 'orange', 'muskmelon', 'banana',
       'Cassava', 'cotton', 'Cassava', 'Potatoes', 'Yam', 'rice', 'jute',
       'mango', 'Plantains', 'Carrot', 'watermelon', 'pigeonpeas', 'Yam',
       'Tomatoes', 'mango', 'jute', 'coffee', 'Carrot', 'Yam', 'coconut',
       'Plantains', 'coffee', 'Potatoes', 'coconut', 'orange', 'cotton',
       'Plantains', 'banana', 'maize', 'muskmelon', 'Tomatoes', 'rice',
       'papaya', 'Tomatoes', 'pigeonpeas', 'muskmelon', 'Yam', 'maize',
       'coffee', 'maize', 'muskmelon', 'banana', 'Plantains', 'mango',
       'Carrot', 'orange', 'banana', 'Tomatoes', 'Pepper', 'jute',
       'watermelon', 'Cassava', 'coffee', 'Cassava', 'papaya', 'banana',
       'papaya', 'Plantains', 'Yam', 'Potatoes', 'banana', 'muskmelon',
       'coffee', 'Carrot', 'coconut', 'coffee', 'coffee', 'Pepper',
       'Carrot', 'orange', 'Potatoes', 'coffee', 'orange', 'Pepper',
       'muskmelon', 'muskmelon', 'Cassava', 'Potatoes', 'cotton',
       'Cassava', 'Yam', 'watermelon', 'papaya', 'orange', 'coconut',
       'Tomatoes', 'Plantains', 'Cassava', 'cotton', 'jute', 'Carrot',
       'Tomatoes', 'Cassava', 'coffee', 'orange', 'rice', 'coffee',
       'mango', 'muskmelon', 'pigeonpeas', 'Cassava', 'banana', 'coconut',
       'watermelon', 'pigeonpeas', 'banana', 'Tomatoes', 'papaya',
       'papaya', 'jute', 'papaya', 'Cassava', 'coconut', 'Yam',
       'Plantains', 'pigeonpeas', 'Tomatoes', 'coffee', 'Pepper',
       'coconut', 'rice', 'Carrot', 'Tomatoes', 'papaya', 'Tomatoes',
       'Plantains', 'cotton', 'Plantains', 'Yam', 'mango', 'cotton',
       'orange', 'Plantains', 'coffee', 'maize', 'Tomatoes', 'coconut',
       'Plantains', 'coffee', 'papaya', 'pigeonpeas', 'Yam', 'Carrot',
       'orange', 'coconut', 'Yam', 'Potatoes', 'cotton', 'cotton',
       'muskmelon', 'mango', 'Carrot', 'Plantains', 'rice', 'Yam',
       'Pepper', 'Carrot', 'cotton', 'Plantains', 'mango', 'Potatoes',
       'papaya', 'jute', 'Cassava', 'cotton', 'papaya', 'papaya',
       'Carrot', 'cotton', 'coffee', 'orange', 'orange', 'maize',
       'Carrot', 'coconut', 'orange', 'rice', 'Tomatoes', 'orange',
       'Pepper', 'papaya', 'cotton', 'coconut', 'Carrot', 'rice',
       'Tomatoes', 'pigeonpeas', 'watermelon', 'cotton', 'Potatoes',
       'Tomatoes'], dtype=object)
crop_names_y_test = label_encoder.inverse_transform(y_test)
crop_names_y_test
array(['jute', 'Plantains', 'muskmelon', 'pigeonpeas', 'watermelon',
       'watermelon', 'pigeonpeas', 'cotton', 'Cassava', 'muskmelon',
       'Cassava', 'watermelon', 'mango', 'Tomatoes', 'Plantains',
       'Potatoes', 'rice', 'Yam', 'coconut', 'jute', 'Pepper', 'coconut',
       'coffee', 'banana', 'coffee', 'Cassava', 'mango', 'pigeonpeas',
       'rice', 'Cassava', 'Pepper', 'pigeonpeas', 'Tomatoes',
       'pigeonpeas', 'banana', 'Carrot', 'coconut', 'watermelon',
       'banana', 'Potatoes', 'Yam', 'coffee', 'Potatoes', 'Tomatoes',
       'papaya', 'maize', 'Tomatoes', 'maize', 'Cassava', 'coconut',
       'papaya', 'Potatoes', 'orange', 'Potatoes', 'pigeonpeas', 'cotton',
       'watermelon', 'coconut', 'mango', 'Plantains', 'orange', 'Pepper',
       'coconut', 'banana', 'coffee', 'jute', 'pigeonpeas', 'papaya',
       'Yam', 'orange', 'orange', 'Plantains', 'muskmelon', 'banana',
       'rice', 'Plantains', 'coconut', 'rice', 'Plantains', 'jute',
       'Cassava', 'maize', 'orange', 'papaya', 'coconut', 'orange',
       'cotton', 'Carrot', 'Potatoes', 'coconut', 'papaya', 'cotton',
       'Potatoes', 'Potatoes', 'jute', 'jute', 'Pepper', 'watermelon',
       'papaya', 'coffee', 'pigeonpeas', 'pigeonpeas', 'Carrot',
       'muskmelon', 'Potatoes', 'maize', 'Pepper', 'Potatoes', 'Pepper',
       'Tomatoes', 'rice', 'orange', 'banana', 'cotton', 'rice',
       'Plantains', 'mango', 'pigeonpeas', 'Plantains', 'coconut', 'jute',
       'banana', 'mango', 'Carrot', 'Plantains', 'papaya', 'Carrot',
       'maize', 'mango', 'Plantains', 'jute', 'jute', 'rice', 'Tomatoes',
       'maize', 'watermelon', 'coffee', 'coffee', 'coconut', 'papaya',
       'muskmelon', 'Plantains', 'coconut', 'maize', 'papaya', 'Tomatoes',
       'Pepper', 'watermelon', 'Potatoes', 'Yam', 'coconut', 'pigeonpeas',
       'Pepper', 'mango', 'Plantains', 'Yam', 'Tomatoes', 'coconut',
       'Tomatoes', 'coconut', 'Pepper', 'maize', 'Yam', 'mango', 'coffee',
       'orange', 'cotton', 'watermelon', 'Potatoes', 'cotton', 'mango',
       'Pepper', 'coffee', 'jute', 'Carrot', 'muskmelon', 'rice',
       'muskmelon', 'maize', 'coconut', 'watermelon', 'rice', 'jute',
       'mango', 'maize', 'rice', 'coffee', 'coconut', 'mango',
       'watermelon', 'Pepper', 'Potatoes', 'muskmelon', 'Potatoes',
       'Cassava', 'Carrot', 'mango', 'maize', 'Pepper', 'papaya', 'jute',
       'Tomatoes', 'Yam', 'rice', 'Carrot', 'Potatoes', 'orange',
       'muskmelon', 'banana', 'Cassava', 'cotton', 'Cassava', 'Potatoes',
       'Yam', 'rice', 'jute', 'mango', 'Plantains', 'Carrot',
       'watermelon', 'pigeonpeas', 'Yam', 'Tomatoes', 'mango', 'jute',
       'coffee', 'Carrot', 'Yam', 'coconut', 'Plantains', 'coffee',
       'Potatoes', 'coconut', 'orange', 'cotton', 'Plantains', 'banana',
       'maize', 'muskmelon', 'Tomatoes', 'rice', 'papaya', 'Tomatoes',
       'pigeonpeas', 'muskmelon', 'Yam', 'maize', 'coffee', 'maize',
       'muskmelon', 'banana', 'Plantains', 'mango', 'Carrot', 'orange',
       'banana', 'Tomatoes', 'Pepper', 'jute', 'watermelon', 'Cassava',
       'coffee', 'Cassava', 'papaya', 'banana', 'papaya', 'Plantains',
       'Yam', 'Potatoes', 'banana', 'muskmelon', 'coffee', 'Carrot',
       'coconut', 'coffee', 'coffee', 'Pepper', 'Carrot', 'orange',
       'Potatoes', 'coffee', 'orange', 'Pepper', 'muskmelon', 'muskmelon',
       'Cassava', 'Potatoes', 'cotton', 'Cassava', 'Yam', 'watermelon',
       'papaya', 'orange', 'coconut', 'Tomatoes', 'Plantains', 'Cassava',
       'cotton', 'jute', 'Carrot', 'Tomatoes', 'Cassava', 'coffee',
       'orange', 'rice', 'coffee', 'mango', 'muskmelon', 'pigeonpeas',
       'Cassava', 'banana', 'coconut', 'watermelon', 'pigeonpeas',
       'banana', 'Tomatoes', 'papaya', 'papaya', 'jute', 'papaya',
       'Cassava', 'coconut', 'Yam', 'Plantains', 'pigeonpeas', 'Tomatoes',
       'coffee', 'Pepper', 'coconut', 'rice', 'Carrot', 'Tomatoes',
       'papaya', 'Tomatoes', 'Plantains', 'cotton', 'Plantains', 'Yam',
       'mango', 'cotton', 'orange', 'Plantains', 'coffee', 'maize',
       'Tomatoes', 'coconut', 'Plantains', 'coffee', 'papaya',
       'pigeonpeas', 'Yam', 'Carrot', 'orange', 'coconut', 'Yam',
       'Tomatoes', 'cotton', 'cotton', 'muskmelon', 'mango', 'Carrot',
       'Plantains', 'rice', 'Yam', 'Pepper', 'Carrot', 'cotton',
       'Plantains', 'mango', 'Potatoes', 'papaya', 'jute', 'Cassava',
       'cotton', 'papaya', 'papaya', 'Carrot', 'cotton', 'coffee',
       'orange', 'orange', 'maize', 'Carrot', 'coconut', 'orange', 'rice',
       'Tomatoes', 'orange', 'Pepper', 'papaya', 'cotton', 'coconut',
       'Carrot', 'rice', 'Tomatoes', 'pigeonpeas', 'watermelon', 'cotton',
       'Potatoes', 'Tomatoes'], dtype=object)
print(isinstance(y_pred, np.ndarray))  # Check if y_pred is a numpy array
print(isinstance(y_test, np.ndarray))  # Check if y_test is a numpy array
True
False
y_test = y_test.to_numpy()
print(isinstance(y_test, np.ndarray))
True
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
No description has been provided for this image
crop_names = label_encoder.classes_


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix with crop names
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=crop_names, yticklabels=crop_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)  # Rotate x labels if necessary
plt.yticks(rotation=0)   # Rotate y labels if necessary
plt.show()
No description has been provided for this image
from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        21
           1       1.00      1.00      1.00        18
           2       1.00      1.00      1.00        18
           3       0.96      0.96      0.96        25
           4       0.92      1.00      0.96        22
           5       1.00      0.92      0.96        25
           6       0.94      0.94      0.94        18
           7       1.00      1.00      1.00        15
           8       1.00      1.00      1.00        28
           9       1.00      1.00      1.00        24
          10       1.00      1.00      1.00        20
          11       1.00      1.00      1.00        18
          12       1.00      1.00      1.00        16
          13       1.00      1.00      1.00        19
          14       1.00      1.00      1.00        17
          15       1.00      1.00      1.00        21
          16       1.00      1.00      1.00        23
          17       1.00      1.00      1.00        18
          18       1.00      1.00      1.00        18
          19       1.00      1.00      1.00        16

    accuracy                           0.99       400
   macro avg       0.99      0.99      0.99       400
weighted avg       0.99      0.99      0.99       400

from sklearn.model_selection import cross_val_score

# Perform cross-validation and calculate average accuracy
cross_val_scores = cross_val_score(rfc, X_train, y_train, cv=5)
print("Cross-validation accuracy scores:", cross_val_scores)
print("Average cross-validation accuracy:", cross_val_scores.mean())
Cross-validation accuracy scores: [0.984375 0.9875   0.984375 0.990625 0.99375 ]
Average cross-validation accuracy: 0.9881249999999999
import joblib

# Save the model as CropRec.pkl
joblib.dump(rfc, 'CropRec.joblib')
['CropRec.joblib']
loaded_model = joblib.load('CropRec.joblib')
joblib.dump(label_encoder, 'CropRec_LabelEncoder.joblib')
['CropRec_LabelEncoder.joblib']
label_encoder = joblib.load('CropRec_LabelEncoder.joblib')
y_pred = loaded_model.predict(X_test)

# Convert the predicted labels to crop names
crop_names = label_encoder.inverse_transform(y_pred)
crop_names
array(['jute', 'Plantains', 'muskmelon', 'pigeonpeas', 'watermelon',
       'watermelon', 'pigeonpeas', 'cotton', 'Cassava', 'muskmelon',
       'Cassava', 'watermelon', 'mango', 'Tomatoes', 'Plantains',
       'Potatoes', 'rice', 'Yam', 'coconut', 'jute', 'Pepper', 'coconut',
       'coffee', 'banana', 'coffee', 'Cassava', 'mango', 'pigeonpeas',
       'rice', 'Cassava', 'Pepper', 'pigeonpeas', 'Tomatoes',
       'pigeonpeas', 'banana', 'Carrot', 'coconut', 'watermelon',
       'banana', 'Potatoes', 'Yam', 'coffee', 'Potatoes', 'Tomatoes',
       'papaya', 'maize', 'Tomatoes', 'maize', 'Cassava', 'coconut',
       'papaya', 'Potatoes', 'orange', 'Potatoes', 'pigeonpeas', 'cotton',
       'watermelon', 'coconut', 'mango', 'Plantains', 'orange', 'Pepper',
       'coconut', 'banana', 'coffee', 'jute', 'pigeonpeas', 'papaya',
       'Yam', 'orange', 'orange', 'Plantains', 'muskmelon', 'banana',
       'rice', 'Yam', 'coconut', 'rice', 'Plantains', 'jute', 'Cassava',
       'maize', 'orange', 'papaya', 'coconut', 'orange', 'cotton',
       'Carrot', 'Potatoes', 'coconut', 'papaya', 'cotton', 'Potatoes',
       'Potatoes', 'jute', 'jute', 'Pepper', 'watermelon', 'papaya',
       'coffee', 'pigeonpeas', 'pigeonpeas', 'Carrot', 'muskmelon',
       'Potatoes', 'maize', 'Pepper', 'Potatoes', 'Pepper', 'Potatoes',
       'rice', 'orange', 'banana', 'cotton', 'rice', 'Plantains', 'mango',
       'pigeonpeas', 'Plantains', 'coconut', 'jute', 'banana', 'mango',
       'Carrot', 'Plantains', 'papaya', 'Carrot', 'maize', 'mango',
       'Plantains', 'jute', 'jute', 'rice', 'Tomatoes', 'maize',
       'watermelon', 'coffee', 'coffee', 'coconut', 'papaya', 'muskmelon',
       'Plantains', 'coconut', 'maize', 'papaya', 'Tomatoes', 'Pepper',
       'watermelon', 'Potatoes', 'Yam', 'coconut', 'pigeonpeas', 'Pepper',
       'mango', 'Plantains', 'Yam', 'Tomatoes', 'coconut', 'Tomatoes',
       'coconut', 'Pepper', 'maize', 'Yam', 'mango', 'coffee', 'orange',
       'cotton', 'watermelon', 'Potatoes', 'cotton', 'mango', 'Pepper',
       'coffee', 'jute', 'Carrot', 'muskmelon', 'rice', 'muskmelon',
       'maize', 'coconut', 'watermelon', 'rice', 'jute', 'mango', 'maize',
       'rice', 'coffee', 'coconut', 'mango', 'watermelon', 'Pepper',
       'Potatoes', 'muskmelon', 'Potatoes', 'Cassava', 'Carrot', 'mango',
       'maize', 'Pepper', 'papaya', 'jute', 'Tomatoes', 'Plantains',
       'rice', 'Carrot', 'Potatoes', 'orange', 'muskmelon', 'banana',
       'Cassava', 'cotton', 'Cassava', 'Potatoes', 'Yam', 'rice', 'jute',
       'mango', 'Plantains', 'Carrot', 'watermelon', 'pigeonpeas', 'Yam',
       'Tomatoes', 'mango', 'jute', 'coffee', 'Carrot', 'Yam', 'coconut',
       'Plantains', 'coffee', 'Potatoes', 'coconut', 'orange', 'cotton',
       'Plantains', 'banana', 'maize', 'muskmelon', 'Tomatoes', 'rice',
       'papaya', 'Tomatoes', 'pigeonpeas', 'muskmelon', 'Yam', 'maize',
       'coffee', 'maize', 'muskmelon', 'banana', 'Plantains', 'mango',
       'Carrot', 'orange', 'banana', 'Tomatoes', 'Pepper', 'jute',
       'watermelon', 'Cassava', 'coffee', 'Cassava', 'papaya', 'banana',
       'papaya', 'Plantains', 'Yam', 'Potatoes', 'banana', 'muskmelon',
       'coffee', 'Carrot', 'coconut', 'coffee', 'coffee', 'Pepper',
       'Carrot', 'orange', 'Potatoes', 'coffee', 'orange', 'Pepper',
       'muskmelon', 'muskmelon', 'Cassava', 'Potatoes', 'cotton',
       'Cassava', 'Yam', 'watermelon', 'papaya', 'orange', 'coconut',
       'Tomatoes', 'Plantains', 'Cassava', 'cotton', 'jute', 'Carrot',
       'Tomatoes', 'Cassava', 'coffee', 'orange', 'rice', 'coffee',
       'mango', 'muskmelon', 'pigeonpeas', 'Cassava', 'banana', 'coconut',
       'watermelon', 'pigeonpeas', 'banana', 'Tomatoes', 'papaya',
       'papaya', 'jute', 'papaya', 'Cassava', 'coconut', 'Yam',
       'Plantains', 'pigeonpeas', 'Tomatoes', 'coffee', 'Pepper',
       'coconut', 'rice', 'Carrot', 'Tomatoes', 'papaya', 'Tomatoes',
       'Plantains', 'cotton', 'Plantains', 'Yam', 'mango', 'cotton',
       'orange', 'Plantains', 'coffee', 'maize', 'Tomatoes', 'coconut',
       'Plantains', 'coffee', 'papaya', 'pigeonpeas', 'Yam', 'Carrot',
       'orange', 'coconut', 'Yam', 'Potatoes', 'cotton', 'cotton',
       'muskmelon', 'mango', 'Carrot', 'Plantains', 'rice', 'Yam',
       'Pepper', 'Carrot', 'cotton', 'Plantains', 'mango', 'Potatoes',
       'papaya', 'jute', 'Cassava', 'cotton', 'papaya', 'papaya',
       'Carrot', 'cotton', 'coffee', 'orange', 'orange', 'maize',
       'Carrot', 'coconut', 'orange', 'rice', 'Tomatoes', 'orange',
       'Pepper', 'papaya', 'cotton', 'coconut', 'Carrot', 'rice',
       'Tomatoes', 'pigeonpeas', 'watermelon', 'cotton', 'Potatoes',
       'Tomatoes'], dtype=object)
       
test_input = [[28.5,	59.11,	22.43,	30.90607799, 52.79913039,	7.05181629,	170.99198280000005]]
column_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Convert test input to DataFrame
test_input_df = pd.DataFrame(test_input, columns=column_names)
y_pred = loaded_model.predict(test_input_df)

# Convert the predicted labels to crop names
crop_names = label_encoder.inverse_transform(y_pred)
crop_names
array(['pigeonpeas'], dtype=object)
 
 
