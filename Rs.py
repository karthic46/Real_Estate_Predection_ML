import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_excel('C:\Users\karth\OneDrive\Documents\GitHub\Real_Estate_predection_ML\real_estate_data.xlsx')

print("Dataset Overview:")
print(data.head())
print("\nData Summary:")
print(data.describe())

X = data[['SquareFootage', 'Bedrooms', 'Bathrooms', 'Age', 'LocationRating']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}

linear_results = evaluate_model(linear_model, X_test, y_test)
print("\nLinear Regression Results:", linear_results)

rf_results = evaluate_model(rf_model, X_test, y_test)
print("\nRandom Forest Results:", rf_results)

y_test = y_test.reset_index(drop=True)
y_pred_rf = rf_model.predict(X_test)
results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred_rf})
print("\nPredicted vs Actual Prices (Random Forest Model):")
print(results.head(10))