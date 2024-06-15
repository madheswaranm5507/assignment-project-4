import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import sklearn
import pickle

# user input options
class options:

    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 
                      77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    
    status_values = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
                     'Wonderful', 'Revised', 'Offered', 'Offerable']
    status_dict = {"Lost":0,"Won":1,"Draft":2,"To be approved":3,"Not lost for AM":4,"Wonderful":5,
                   "Revised":6,"Offered":7,"Offerable":8}
    
    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR','Others']
    item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 
                          29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    
    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407, 164337175, 929423819, 
                          1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
                          1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]



# Functions in Predict_status("Won" or "Lost")

def predict_status(country,item_type,application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                   selling_price_log,item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                    delivery_date_year):
    
    # pickle file for Classification Model
    with open(r"C:\Users\madhe\OneDrive\Desktop\Industrial Copper Modeling\Classification_model.pkl", "rb") as f:
        model_class = pickle.load(f)

    user_class_data = np.array([[country,options.item_type_dict[item_type],application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                   selling_price_log,item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                    delivery_date_year]])
    
    y_pred = model_class.predict(user_class_data)

    if y_pred == 1:
        return 1
    else:
        return 0
    

# Function in predict_selling_price

def predict_selling_price(country,status,item_type,application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                   item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                    delivery_date_year):
    
    # pickle file for Regression Model
    with open (r"C:\Users\madhe\OneDrive\Desktop\Industrial Copper Modeling\Regression_Model.pkl","rb") as f:
        model_regression = pickle.load(f)

        user_regression_data = np.array([[country,options.status_dict[status],options.item_type_dict[item_type],application,width,product_ref,quantity_tons_log,customer_log,thickness_log,
                   item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,
                    delivery_date_year]])
        
        y_pred = model_regression.predict(user_regression_data)

        exponential_y_pred = np.exp(y_pred[0])

        return exponential_y_pred


# Streamlit Part
st.set_page_config(layout="wide")
st.title(":red[**INDUSTRIAL COPPER MODELING**]")

with st.sidebar:
    option = option_menu("DATA EXPLORATION", options=["HOME", "PREDICT SELLING PRICE", "PREDICT STATUS"])

if option == "HOME":
    
    st.header("Introduction")
    st.write(""" 
             In the copper industry, dealing with complex sales and pricing data can be challenging. 
             Our solution employs advanced machine learning techniques to address these challenges, offering regression models for precise pricing predictions and lead classification for better customer targeting. 
             I also gain experience in data preprocessing, feature engineering, and web application development using Streamlit, equipping you to solve real-world problems in manufacturing.""")
    
    st.header("Technologies Used")
    st.write("1. Python - This project is implemented using the Python programming language.")
    st.write("2. Pandas - A Powerfull data manipulation in pandas. providing functionalities such as data filtering, dataframe create, transformation, and aggregation.")
    st.write("3. Numpy - Is an essential library for numerical computations in Python, offering a vast array of functionalities to manipulate and operate on arrays and matrices efficiently.")
    st.write("4. Scikit-Learn - This one of the most popular libraries for machine learning in Python. offering a wide range of supervised and unsupervised machine learning algorithms.")
    st.write("5. Matplotlib - A wide range of plot types including line plots, scatter plots, bar plots, histograms, pie charts, and more. It also supports complex visualizations like 3D plots, contour plots, and image plots.")
    st.write("6. Seaborn - It provides a high-level interface for drawing attractive and informative statistical graphics.")
    st.write("7. Pickle - A useful Python tool that allows to save the ML models, to minimise lengthy re-training and allow to share, commit, and re-load pre-trained machine learning models")
    st.write("8. Streamlit - The user interface and visualization are created using the Streamlit framework.")



if option == "PREDICT SELLING PRICE":

    st.header(":green[PREDICT SELLING PRICE]")
    st.write("")

    col1,col2 = st.columns(2)
    with col1:
        country = st.selectbox(label='Country', options=options.country_values)

        status = st.selectbox(label='Status', options=options.status_values)

        item_type = st.selectbox(label='Item Type', options=options.item_type_values)

        application = st.selectbox(label='Application', options=options.application_values)

        product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)

        quantity_tons_log = st.number_input(label='Quantity Tons Log Value [Min:-11.51 & Max:20.72]', format= "%0.2f")

        customer_log = st.number_input(label='Customer Log Value [Min:9.43 & Max:21.48]', format="%0.2f")

        thickness_log = st.number_input(label="Thickness Log Value [Min:-1.71 & Max:7.82]", format="%0.2f")
       
        
    with col2:

        width = st.number_input(label='Width [Min:1.0 & Max:2990.0]')

        item_date_day = st.number_input(label='Item Date Day [Min:1 & Max:31]')

        item_date_month = st.number_input(label='Item Date Month [Min:1 & Max:12]')

        item_date_year = st.number_input(label='Item Date Year [Min:2020 & Max:2021]')

        delivery_date_day = st.number_input(label='Delivery Date Day [Min:1 & Max:31]')

        delivery_date_month = st.number_input(label='Delivery Date Month [Min:1 & Max:12]')

        delivery_date_year = st.number_input(label='Delivery Date Year [Min:2020 & Max:2022]')
      
  
    
    button = st.button(":blue[PREDICT THE SELLING PRICE]", use_container_width=True)

    if button:
        price = predict_selling_price(country,status,item_type,application,product_ref,quantity_tons_log,customer_log,thickness_log,
                                      width,item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,delivery_date_year)
        
        st.write(":green[THE SELLING PRICE IS : ]",price)



if option == "PREDICT STATUS":

    st.header(":green[PREDICT THE STATUS (WON / LOST)]")
    st.write(" ")
    
    col1,col2 = st.columns(2)
    with col1:
       
        country = st.selectbox(label='Country', options=options.country_values)

        item_type = st.selectbox(label='Item Type', options=options.item_type_values)

        application = st.selectbox(label='Application', options=options.application_values)

        product_ref = st.selectbox(label='Product Ref', options=options.product_ref_values)

        quantity_tons_log = st.number_input(label='Quantity Tons Log Value [Min:-11.51 & Max:20.72]', format= "%0.2f")

        customer_log = st.number_input(label='Customer Log Value [Min:9.43 & Max:21.48]', format="%0.2f")

        thickness_log = st.number_input(label='Thickness Log Value [Min:-1.71 & Max:7.82]', format="%0.2f")

        selling_price_log = st.number_input(label= 'Selling Price Log Value [Min:-2.30 & Max:18.42]',format="%0.2f")

    with col2:

        width = st.number_input(label='Width [Min:1.0 & Max:2990.0]')

        item_date_day = st.number_input(label='Item Date Day [Min:1 & Max:31]')

        item_date_month = st.number_input(label='Item Date Month [Min:1 & Max:12]')

        item_date_year = st.number_input(label='Item Date Year [Min:2020 & Max:2021]')

        delivery_date_day = st.number_input(label='Delivery Date Day [Min:1 & Max:31]')

        delivery_date_month = st.number_input(label='Delivery Date Month [Min:1 & Max:12]')

        delivery_date_year = st.number_input(label='Delivery Date Year [Min:2020 & Max:2022]')
     
  
    
    button = st.button(":violet[PREDICT THE STATUS]" ,use_container_width=True)
    st.header(":orange[THE STATUS IS WON / LOST]")
    if button:
        status = predict_status(country,item_type,application,product_ref,quantity_tons_log,customer_log,thickness_log,selling_price_log,
                                width,item_date_day,item_date_month,item_date_year,delivery_date_day,delivery_date_month,delivery_date_year)
        
       
        if status == 1:
            st.write(":green[THE STATUS IS WON]")
        else:
            st.write(":blue[THE STATUS IS LOST]")        
 
    
    st.write(" ") # space mention
    st.write(" ") # space mention
    st.header("Conclusion")
    st.write("""This project provides accurate predictions for both the product's selling price and status through the implementation of various machine learning techniques. 
             With its user-friendly interface, it offers an accessible environment for users to interact with and make informed decisions based on the model's outputs.""")





    

















