import pandas as pd
import numpy as np

train_df = pd.read_csv('paris_airbnb_train.csv')
test_df = pd.read_csv('paris_airbnb_test.csv')

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [1,2,3,4,5]
mse_values = list()

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
    
print(mse_values)

#plus k augmente, plus le modèle s'améliore

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [x for x in range(1,60)]
mse_values = list()

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
    
print(mse_values)

print(min(mse_values))

import matplotlib.pyplot as plt


features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [x for x in range(9,300)]
mse_values = list()

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)


plt.scatter(hyper_params, mse_values)
plt.show()

hyper_params = [x for x in range(1,100)]
mse_values = list()
features = train_df.columns.tolist()
features.remove('price')

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)


plt.scatter(hyper_params, mse_values)
plt.show()

hyper_params = [x for x in range(1,100)]
mse_values = list()
features = train_df.columns.tolist()
features.remove('price')

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)


plt.scatter(hyper_params, mse_values)
plt.show()

print(min(mse_values))

#pratiquer le WorkFlow

two_hp_mse = dict()
three_hp_mse = dict()

features2 = ['accommodates', 'bathrooms']
hyper_params = [x for x in range(1,100)]
two_mse_values = list()

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features2], train_df['price'])
    predictions = knn.predict(test_df[features2])
    mse = mean_squared_error(test_df['price'], predictions)
    two_mse_values.append(mse)
    
two_lowest_mse = two_mse_values[0]
two_lowest_k = 1

for k, mse in enumerate(two_mse_values):
    if mse < two_lowest_mse :
        two_lowest_mse = mse
        two_lowest_k = k + 1
        
two_hp_mse[two_lowest_k] = two_lowest_mse
    
print(two_hp_mse)

features3 = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,100)]
three_mse_values = list()

for i in hyper_params :
    knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
    knn.fit(train_df[features3], train_df['price'])
    predictions = knn.predict(test_df[features3])
    mse = mean_squared_error(test_df['price'], predictions)
    three_mse_values.append(mse)
    
three_lowest_mse = three_mse_values[0]
three_lowest_k = 1

for k, mse in enumerate(three_mse_values):
    if mse < three_lowest_mse :
        three_lowest_mse = mse
        three_lowest_k = k + 1
        
three_hp_mse[three_lowest_k] = three_lowest_mse
    
print(three_hp_mse)

