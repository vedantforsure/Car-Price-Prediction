import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("car_prices_prediction.csv")

print(df.head())

# Select independent and dependent variable
x = df[['make', 'model', 'year', 'mileage']] 
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
reg = RandomForestRegressor(n_estimators=1, random_state=0).fit(X_train,y_train)
y_pred = reg.predict(X_test)
# Split the dataset into train and test




# # Instantiate the model
# model = RandomForestRegressor()

# # Fit the model
# model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(reg, open("car_price_pred", "wb"))