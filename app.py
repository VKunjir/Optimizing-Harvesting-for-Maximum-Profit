import numpy as np
import pickle4 as pickle 
import pandas as pd
import calendar

import streamlit as st

# Load the trained regressor model
pickle_in = open("regressor.pkl", "rb")
regressor = pickle.load(pickle_in)

# Load the fitted ColumnTransformer
column_transformer = pickle.load(open("column_transformer.pkl", "rb"))

# Define the list of agricultural goods
agricultural_goods_list = [
    "Aale", "Aalu", "Aboli", "Amaranth", "Anthurium", "Apple", "Aster", "AshGourd", "AsterDhakli", "Avocado",
    "Babycorn", "Banana", "Basil", "Bathuaa", "Beetroot", "BhavnagriChilli", "Bijali", "BitterMelon", "Blueberry",
    "BottleGourd", "BroadBeans", "Broccoli", "Cabbage", "Capsicum", "Carelessweed", "Carnation", "Carrot",
    "Cauliflower", "Celery", "Chafa", "Chavalai", "Cherry", "CherryTomato", "Chickpeas", "Chili", "ChinaCabbage",
    "ChineseGarlic", "ChrysanthemumKaadi", "Clitoriaternatea", "ClusterBean", "Cockscomb", "Coconut", "ColoredGladiolus",
    "Coriander", "Cowpea", "Cucumber", "CurryLeaf", "CustardApple", "Dangar", "Date", "Dhana", "Dill", "DoubleBee",
    "DragonFruit", "Drumstick", "DutchRose", "Eggplant", "Fenugreek", "FieldBeans", "Fig", "Filler", "FingerMillet",
    "Frangipani", "FrangipaniKaadi", "Gaillardia", "Garlic", "Gerbera", "Ghewada", "GoldenRed", "Gooseberry",
    "Grapes", "GreenCoconut", "GreenGram", "GreenMango", "GreenPeas", "GreenSorrel", "Guava", "Gultop", "Gypsy",
    "HGaddi", "Hibiscus", "HorseGram", "Iceberg", "IvyGourd", "Jackfruit", "Jaggery", "Jui", "Jujube", "Kagda",
    "Kamini", "Kiwifruit", "Ladyless", "Lemon", "Lentils", "Lilium", "Lily", "LimaBeans", "Litchi", "LittleMillet",
    "LotusRoot", "Maize", "Mogra", "Mushroom", "Muskmelon", "Mustard", "Okra", "OnionLeaf", "Orange", "Orchid",
    "Papaya", "Parcel", "Peach", "Peanut", "Pear", "Peas", "Pineapple", "Plum", "PointedGourd", "Pokcha", "Pomegranate",
    "Potato", "Radish", "RedCabbage", "Rice", "RidgeGourd", "Roman", "RoseGladiator", "Roselle", "Sacurry", "Safflower",
    "Salad", "Sapodilla", "Sapota", "SewgaSheng", "SimpleGladiolus", "SimpleRose", "SnakeGourd", "Sorghum", "Spinach",
    "Springeri", "Strawberry", "Suran", "SuttaAster", "Sweetcorn", "SweetLemon", "SweetPotato", "Tamarind", "TaroRoot",
    "Tinda", "Tomato", "TukdaRose", "TuljapuriMarigold", "Turmeric", "Watermelon", "Wheat", "WhiteChrysanthemum",
    "WholeBlackGram", "WoodApple", "YellowChrysanthemum", "Zucchini"
]

def predict_top_prices(agriculturalGoods, year, top_n):
    year_ = pd.DataFrame({'AgriculturalGoods': [agriculturalGoods] * 12, 'Month': range(1, 13), 'Year': [year] * 12})
    year_encoded = column_transformer.transform(year_)
    predicted_prices = regressor.predict(year_encoded)
    top_n = min(top_n, 12)
    top_prices_indices = np.argsort(predicted_prices[:, 1])[::-1][:top_n]
    top_prices = predicted_prices[top_prices_indices]
    top_months = top_prices_indices + 1
    return top_prices, top_months

def main():
    st.title("Optimizing Harvesting for Maximum Profit")
    st.write("Enter the agricultural good, the year, and the number of top prices to predict.")
    
    agriculturalGoods = st.selectbox("Agricultural Good", agricultural_goods_list)  # Input for crop name
    year = st.select_slider("Year", options=range(2021, 2026))  # Input for year
    top_n = st.slider("Number of Top Prices", min_value=1, max_value=12, value=5)  # Slider for selecting top prices

    if st.button("Predict"):
        top_prices, top_months = predict_top_prices(agriculturalGoods, year, top_n)
        st.write(f"The top {top_n} months for harvesting {agriculturalGoods} in {year} are:")
        for month, price in zip(top_months, top_prices):
            month_name = calendar.month_name[month]
            st.write(f"{month_name}: Max Price={price[1]}, Min Price={price[0]}")


if __name__ == '__main__':
    main()
