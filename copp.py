#Numpy Library
import numpy as np
#pickle library to load ML model
import pickle
#Dashboard Libraries
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# Configuring Streamlit GUI

st.set_page_config(layout="wide")

#Menu options

selected = option_menu(None,
                           options = ["Home","Status Prediction","Price Prediction","Machine Learning"],
                           icons = ["house","trophy","currency-rupee","archive"],
                           default_index=0,
                           orientation="horizontal",
                           styles={"container": {"width": "100%"},
                                   "icon": {"color": "white", "font-size": "24px"},
                                   "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                                   "nav-link-selected": {"background-color": "#6F36AD"}})

# # # MENU 1 - Home
if selected == "Home":
    col1,col2 = st.columns(2)
    with col1:
        st.write("The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions")
        st.write("A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data")
        st.header("Tools and Technologies Used")
        st.write("Python,Streamlit,Numpy,Pandas,Scikit-learn,Matplotlib,Pickle,Seaborn")
        with col2:
            st.image("coppermod.jpg")

# # # MENU 2 - Status Prediction

 # Define the possible values for the dropdown menus
if selected=="Status Prediction":
    status = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    status_dict = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}
    item_type = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    #item_type = [5.0, 6.0, 3.0,1.0,2.0,0.0,4.0]
    item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product_ref=['611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                 '611993', '611733','929423819', '1282007633', '1332077137', '164141591', '164336407', 
                 '164337175', '1665572032', '1665572374', '1665584320', '1665584642', 
                 '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                 '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    # Define the widgets for user input
    #form-takes all the inputs and process with it further
    with st.form("my_form"):
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.write("")
            item_type = st.selectbox("Item Type", sorted(item_type),key=2)
            country = st.selectbox("Country", sorted(country_options),key=3)
            application = st.selectbox("Application", sorted(application_options),key=4)
            product_ref = st.selectbox("Product Reference", product_ref,key=5)
        with col3:               
            st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
            quantity_tons = st.text_input("Enter Quantity Tons (Min:0.00001 & Max:10000000)")
            thickness = st.text_input("Enter thickness (Min:0.1 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("Customer ID (Min:12458, Max:30408185)")
            selling_price = st.text_input("Selling Price (Min:1, Max:100001015)") 
            submit_button = st.form_submit_button(label="PREDICT STATUS")
            
        if submit_button:
            try:
                import pickle
                # load the classification pickle model
                with open('classification_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                user_data=np.array([[customer,country,item_type_dict[item_type],application,width,product_ref,np.log(float(quantity_tons)), 
                                    np.log(float(thickness)),np.log(float(selling_price))]])
                # user_data = np.array([[30223403, 78, 5, 10, 1500, 1668701718, 2.2, 0, 7.13]])
                 # model predict the status based on user input
                y_pred = model.predict(user_data)
                status = y_pred[0]
                if status==1:
                    st.write('## :green[The Status is Won] ')
                    st.balloons()
                else:
                    st.write('## :red[The status is Lost] ')
                    st.snow()
            
            except ValueError:
                st.warning('##### Quantity Tons / Customer ID is empty')


# # # MENU 2 - Price Prediction

 # Define the possible values for the dropdown menus
if selected=="Price Prediction":
    status = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    status_dict = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}
    item_type = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    #item_type = [5.0, 6.0, 3.0,1.0,2.0,0.0,4.0]
    item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product_ref=['611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                 '611993', '611733','929423819', '1282007633', '1332077137', '164141591', '164336407', 
                 '164337175', '1665572032', '1665572374', '1665584320', '1665584642', 
                 '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                 '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    # Define the widgets for user input
    with st.form("my_form"):
        col1,col2,col3=st.columns([5,2,5])
        with col1:
            st.write("")
            status = st.selectbox("Status", status,key=1)
            item_type = st.selectbox("Item Type", sorted(item_type),key=2)
            country = st.selectbox("Country", sorted(country_options),key=3)
            application = st.selectbox("Application", sorted(application_options),key=4)
            product_ref = st.selectbox("Product Reference", product_ref,key=5)
        with col3:               
            st.write( f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
            quantity_tons = st.text_input("Enter Quantity Tons (Min:0.00001 & Max:10000000)")
            thickness = st.text_input("Enter thickness (Min:0.1 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("Customer ID (Min:12458, Max:30408185)")
            #selling_price = st.text_input("Selling Price (Min:1, Max:100001015)") 
            submit_button = st.form_submit_button(label="PRICE PREDICTION")
            
        if submit_button:
            try:
                import pickle
                # load the regression pickle model
                with open('regression_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                user_data=np.array([[customer,country,status_dict[status],item_type_dict[item_type],application,width,product_ref,np.log(float(quantity_tons)), 
                                    np.log(float(thickness))]])
                # model predict the price based on user input
                y_pred = model.predict(user_data)
                # inverse transformation for log transformation data
                selling_price = np.exp(y_pred[0])
                
                # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
                selling_price = round(selling_price, 2)
                
                st.write('## :green[Predicted selling price:] ',selling_price)


            except ValueError:
                st.warning('##### Quantity Tons / Customer ID / Selling Price is empty')

# # # MENU 4 - Machine Learning
if selected == "Machine Learning":
    col1,col2=st.columns(2)
    with col1:
        st.write("Machine Learning is the field of study that gives computers the capability to learn without being explicitly programmed.Machine Learning is a branch of artificial intelligence that develops algorithms by learning the hidden patterns of the datasets used it to make predictions on new similar type data, without being explicitly programmed for each task.")
        st.write("Machine learning is used in many different applications, from image and speech recognition to natural language processing, recommendation systems, fraud detection, portfolio optimization, automated task, and so on. ")
        st.write("Types of Machine Learning :1)Supervised Machine Learning   2)Unsupervised Machine Learning   3)Reinforcement Machine Learning")
        st.write("There are two main types of supervised learning:")
        st.write("1.Regression: Regression is a type of supervised learning where the algorithm learns to predict continuous values based on input features.The output labels in regression are continuous values.")
        st.write("2.Classification: Classification is a type of supervised learning where the algorithm learns to assign input data to a specific category or    class based on input features.The output labels in classification are discrete values.")
                          

    with col2:
        st.image("ml.png")
    
    