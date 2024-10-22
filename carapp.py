import streamlit as st
import joblib
import pandas as pd
import category_encoders as ce



# Load the model, scaler, and encoder
rf_model = joblib.load(r'rf_model.pkl')
scaler = joblib.load(r'scaler.pkl')
encoder = joblib.load(r'encoder.pkl')

# User interface
st.title("Car Price Prediction")
st.write("Please enter the features of the car to get a price prediction.")

# Input features
Day = st.selectbox("day", options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
       '24', '25', '26', '27', '28', '29', '30', '31'])

Month = st.selectbox("month", options=['January', 'February', 'March', 'April', 'May', 'June', 'July',
       'August', 'September', 'October', 'November', 'December'])

weekday = st.selectbox("WeekDay'", options=['Saturday','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday' ])

gender = st.selectbox("Gender", options=['Male', 'Female'])

# Annual income input using slider
annual_income = st.slider("Annual Income", min_value=10080, max_value=11200000, value=50000)

dealer_Name = st.selectbox("Dealer_Name", options=["Buddy Storbeck's Diesel Service Inc", 'C & M Motors Inc',
       'Capitol KIA', 'Chrysler of Tri-Cities', 'Chrysler Plymouth',
       'Classic Chevy', 'Clay Johnson Auto Sales', 'U-Haul CO',
       'Rabun Used Car Sales', 'Race Car Help', 'Saab-Belle Dodge',
       'Scrivener Performance Engineering', 'Diehl Motor CO Inc',
       'Star Enterprises Inc', 'Suburban Ford', 'Tri-State Mack Inc',
       'Progressive Shippers Cooperative Association No',
       'Ryder Truck Rental and Leasing', 'Enterprise Rent A Car',
       'Gartner Buick Hyundai Saab', 'Hatfield Volkswagen',
       'Iceberg Rentals', 'McKinney Dodge Chrysler Jeep',
       'Motor Vehicle Branch Office', 'Nebo Chevrolet',
       'New Castle Ford Lincoln Mercury', 'Pars Auto Sales',
       'Pitre Buick-Pontiac-Gmc of Scottsdale']
)
company = st.selectbox("Company", options=['Ford', 'Dodge', 'Cadillac', 'Toyota', 'Acura', 'Mitsubishi',
       'Chevrolet', 'Nissan', 'Mercury', 'BMW', 'Chrysler', 'Subaru',
       'Hyundai', 'Honda', 'Infiniti', 'Audi', 'Porsche', 'Volkswagen',
       'Buick', 'Saturn', 'Mercedes-B', 'Jaguar', 'Volvo', 'Pontiac',
       'Lincoln', 'Oldsmobile', 'Lexus', 'Plymouth', 'Saab', 'Jeep'])

model_input = st.selectbox("Model", options=[
    'Expedition', 'Durango', 'Eldorado', 'Celica', 'TL', 'Diamante', 'Corolla',
    'Galant', 'Malibu', 'Escort', 'RL', 'Pathfinder', 'Grand Marquis', '323i',
    'Sebring Coupe', 'Forester', 'Accent', 'Land Cruiser', 'Accord', '4Runner',
    'I30', 'A4', 'Carrera Cabrio', 'Jetta', 'Viper', 'Regal', 'LHS', 'LW', '3000GT',
    'SLK230', 'Civic', 'S-Type', 'S40', 'Mountaineer', 'Park Avenue',
    'Montero Sport', 'Sentra', 'S80', 'Lumina', 'Bonneville', 'C-Class', 'Altima',
    'DeVille', 'Stratus', 'Cougar', 'SW', 'C70', 'SLK', 'Tacoma', 'M-Class', 'A6',
    'Intrepid', 'Sienna', 'Eclipse', 'Contour', 'Town car', 'Focus', 'Mustang',
    'Cutlass', 'Corvette', 'Impala', 'Cabrio', 'Dakota', '300M', '328i', 'Bravada',
    'Maxima', 'Ram Pickup', 'Concorde', 'V70', 'Quest', 'ES300', 'SL-Class',
    'Explorer', 'Prizm', 'Camaro', 'Outback', 'Taurus', 'Cavalier', 'GS400',
    'Monte Carlo', 'Sonata', 'Sable', 'Metro', 'Voyager', 'Cirrus', 'Avenger',
    'Odyssey', 'Intrigue', 'Silhouette', '5-Sep', '528i', 'LS400', 'Aurora',
    'Breeze', 'Beetle', 'Elantra', 'Continental', 'RAV4', 'Villager', 'S70', 'LS',
    'Ram Van', 'S-Class', 'E-Class', 'Grand Am', 'SC', 'Passat', 'Xterra',
    'Frontier', 'Crown Victoria', 'Camry', 'Navigator', 'CL500', 'Escalade', 'Golf',
    'Ranger', 'Prowler', 'Windstar', 'GTI', 'Passport', 'Boxter', 'LX470', 'CR-V',
    'Sunfire', 'Caravan', 'Ram Wagon', 'Neon', 'Wrangler', 'Integra', 'Grand Prix',
    'Grand Cherokee', 'F-Series', 'A8', 'Mystique', '3-Sep', 'Cherokee',
    'Carrera Coupe', 'Catera', 'Seville', 'CLK Coupe', 'LeSabre', 'Sebring Conv.',
    'GS300', 'Firebird', 'V40', 'Montero', 'Town & Country', 'SL', 'Alero', 'Mirage',
    'Century', 'RX300', 'Avalon'
])
engine = st.selectbox("Engine Type", options=['Double Overhead Camshaft', 'Overhead Camshaft'])
transmission = st.selectbox("Transmission", options=['Auto', 'Manual'])
color = st.selectbox("Color", options=['Black', 'Red', 'Pale White'])

# Adding Body Style option
body_style = st.selectbox("Body Style", options=['SUV', 'Passenger', 'Hatchback', 'Hardtop', 'Sedan'])

# Adding dealer region option
dealer_region = st.selectbox("Dealer Region", options=[
    'Middletown', 'Aurora', 'Greenville', 'Pasco', 'Janesville', 'Scottsdale', 'Austin'
])


# On button click for prediction
if st.button("Predict"):
    # Preparing data for prediction
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Annual Income': [annual_income],
        'Dealer_Name': [dealer_Name],
        'Company': [company],
        'Model': [model_input],
        'Engine': [engine],
        'Transmission': [transmission],
        'Color': [color],
        'Body Style': [body_style],
        'Dealer_Region': [dealer_region],
        'month': [Month],
        'WeekDay': [weekday],
        'day': [Day]

    })

    Binary_encoded = encoder.transform(input_data[['Gender','Dealer_Name', 'Company', 'Model', 'Color', 'Body Style','Dealer_Region','month','WeekDay','day']])
    input_data.drop(columns=['Gender','Dealer_Name', 'Company', 'Model', 'Color', 'Body Style','Dealer_Region','month','WeekDay','day'],inplace=True)
    input_data =pd.concat([input_data,Binary_encoded],axis =1)

    engine_map = {
        'Overhead Camshaft': 1,
        'DoubleÂ\xa0Overhead Camshaft': 2
    }
    input_data['Engine'] = input_data['Engine'].map(engine_map)

    trans_map = {
        'Manual': 1,
        'Auto': 2
    }
    input_data['Transmission'] = input_data['Transmission'].map(trans_map)
    input_data['Transmission'].head()

    input_data['Annual Income'] = scaler.transform(input_data[['Annual Income']])




    # Making prediction
    predicted_price = rf_model.predict(input_data)
    st.write(f"Predicted Price: ${predicted_price[0]:,.2f}")


