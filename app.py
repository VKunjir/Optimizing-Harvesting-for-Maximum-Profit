import numpy as np
import pickle 
import pandas as pd
import calendar
import time

import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import joblib

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .main {
        background-color: #0d0d0d;
        color: white;
    }
    .stSlider > div > div > div > div {
        color: white;
    }
    .css-18e3th9 {
        color: white;
    }
    .st-cg {
        background-color: #0d0d0d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained regressor model
with open("regressor.pkl", "rb") as pickle_in:
    regressor = pickle.load(pickle_in)

# Load the fitted ColumnTransformer
column_transformer = joblib.load("column_transformer.pkl")

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
    # Create a DataFrame with the specified agricultural good, year, and months
    year_data = {
        'AgriculturalGoods': [agriculturalGoods] * 12,
        'Month': range(1, 13),
        'Year': [year] * 12
    }
    year_df = pd.DataFrame(year_data)
    
    # Print input data (for debugging)
    print("Input data (year_df):")
    print(year_df)
    
    try:
        # Transform the input data using the pre-fitted ColumnTransformer
        year_encoded = column_transformer.transform(year_df)
        
        # Predict prices using the trained regressor
        predicted_prices = regressor.predict(year_encoded)
        
        # Get top prices and months
        top_n = min(top_n, 12)
        top_prices_indices = np.argsort(predicted_prices[:, 1])[::-1][:top_n]
        top_prices = predicted_prices[top_prices_indices]
        top_months = top_prices_indices + 1
        
        return top_prices, top_months
    
    except Exception as e:
        # Print exception message (for debugging)
        print("Error during prediction:", e)
        return None, None


def main():
    st.title("Optimizing Harvesting for Maximum Profit")
    st.write("Enter the agricultural good, the year, and the number of months for price prediction.")
    
    agriculturalGoods = st.selectbox("Select agricultural good:", agricultural_goods_list)  # Input for crop name
    year = st.select_slider("Select the year for prediction:", options=range(2021, 2026), value=2024)  # Input for year
    top_n = st.slider("Select the number of months for price prediction:", min_value=1, max_value=12, value=5)  # Slider for selecting top prices

    if st.button("Predict"):
        # Display spinner while waiting for prediction results
        with st.spinner("Predicting..."):
            # Simulate delay (remove this line in actual implementation)
            time.sleep(5)
            
            top_prices, top_months = predict_top_prices(agriculturalGoods, year, top_n)
        
        if top_months is not None and top_prices is not None:
            st.write(f"The top {top_n} months for harvesting {agriculturalGoods} in {year} are:")
            
            # Prepare data for table
            table_data = {
                "Month": [calendar.month_name[month] for month in top_months],
                "Max Price": [f"{price[1]:,.1f}" for price in top_prices],
                "Min Price": [f"{price[0]:,.1f}" for price in top_prices]
            }
            df = pd.DataFrame(table_data)
            st.table(df)
        else:
            st.error("Error occurred during prediction. Please try again.")

if __name__ == '__main__':
    main()