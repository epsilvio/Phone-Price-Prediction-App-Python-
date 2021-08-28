#import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

mobile = pd.read_csv('train.csv')
x = mobile.iloc[:, [0, 2, 3, 4, 5, 6, 10, 13, 18, 19]]
y = mobile['price_range']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)

linreg = LinearRegression()
linreg.fit(x_train, y_train)

test_accuracy = round((linreg.score(x_test, y_test) * 100), 2)
train_accuracy = round((linreg.score(x_train, y_train) * 100), 2)

print('\nAccuracy on test set: ', test_accuracy, '%')
print('Accuracy on train set: ', train_accuracy, '%')

#Displaying the error rate of prediction
#y_pred = linreg.predict(x_test)
#error = pd.DataFrame(np.array(y_test).flatten(), columns=['actual'])
#error['y_pred'] = np.array(y_pred)
#print(error.head(20))

print("\nSimple Phone Price Prediction App (Recommended for Android phones only!)\nThis app uses Linear Regression to predict the phone class and price based on the specs!\n")

def user_input_specs():
    battery = int(input("Batter Capacity (mAh): "))
    clock_speed = float(input("Clock Speed (GHz): "))
    dual_sim = int(input("Dual Sim? No = 0, Yes = 1: "))
    fcam = int(input("Front Camera Quality (Megapixels): "))
    four_g = int(input("Supports 4G? No = 0, Yes = 1: "))
    rom = int(input("Internal Memory Size (GB): "))
    pcam = int(input("Primary Camera Quality (Megapixels): "))
    ram = int(input("RAM Size (MB): "))
    t_screen = int(input("Touchscreen? No = 0, Yes = 1: "))
    wifi = int(input("Has WiFi? No = 0, Yes = 1: "))
    data = {'battery_power': battery,
            'clock_speed': clock_speed,
            'dual_sim': dual_sim,
            'fc': fcam,
            'four_g': four_g,
            'int_memory': rom,
            'pc': pcam,
            'ram': ram,
            'touch_screen': t_screen,
            'wifi': wifi}
    specs = pd.DataFrame(data, index=[0])
    return specs

df = user_input_specs()

#print('Your Phone Specs:')
#print(df)

prediction = linreg.predict(df)
class_index = round(prediction[0])

if class_index < 0:
    class_index = 0

if class_index > 3:
    class_index = 3

phone_class = ["Lower Class", "Lower-Middle Class", "Upper-Middle Class", "Upper Class"]
#Lower Class  = Below P3000, Lower-Middle Class = P3000-7999, Upper-Middle Class = P8000 - 13000, Upper Class = Above P13000

def class_prices(class_index):
    switcher = {
        0: "Below P3000",
        1: "P3000-7999",
        2: "P8000-13000",
        3: "Above P13000",
    }
    return switcher.get(class_index)

estimated_price = class_prices(class_index)
print('\nPrediction: Your phone is', phone_class[class_index], 'or around ', estimated_price)
