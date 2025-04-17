import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.title("Car MPG Predictor")
st.write("""
This app predicts the miles per gallon (MPG) of a car based on input values.
""")

col1, col2 = st.columns(2)

with col1:
    cylinders = st.selectbox(
        "Number of Cylinders",
        options=[3, 4, 5, 6, 8, 12],
        index=1
    )
    
    displacement = st.slider(
        "Engine Displacement (cubic inches)",
        min_value=50.0,
        max_value=500.0,
        value=200.0,
        help="Engine displacement in cubic inches"
    )
    
    horsepower = st.slider(
        "Horsepower",
        min_value=50.0,
        max_value=300.0,
        value=150.0
    )
    
    weight = st.slider(
        "Vehicle Weight (lbs)",
        min_value=1500.0,
        max_value=5000.0,
        value=3000.0
    )

with col2:
    acceleration = st.slider(
        "Acceleration (sec to 60mph)",
        min_value=8.0,
        max_value=25.0,
        value=15.0
    )
    
    model_year = st.slider(
        "Model Year",
        min_value=70,
        max_value=85,
        value=75,
        help="Model year (70 = 1970, 85 = 1985, etc.)"
    )
    
    origin = st.selectbox(
        "Origin",
        options=[
            (1, "American"),
            (2, "European"),
            (3, "Japanese")
        ],
        format_func=lambda x: x[1],
        index=0,
        help="Region where the car was manufactured"
    )


if st.button("Predict MPG", type="primary"):

    payload = {
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight": weight,
        "acceleration": acceleration,
        "model_year": model_year,
        "origin": origin[0]  # Extract the numeric code
    }
    
    with st.spinner("Predicting MPG..."):
        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            predicted_mpg = result["predicted_mpg"]
            
            st.success(f"Predicted MPG: {predicted_mpg:.2f}")
            
            # Create a gauge chart using plotly
            fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_mpg,
                    title={"text": "Predicted Miles Per Gallon"},
                    gauge={
                        "axis": {"range": [5, 50]},
                        "bar": {"color": "green"},
                        "steps": [
                            {"range": [5, 15], "color": "red"},
                            {"range": [15, 25], "color": "yellow"},
                            {"range": [25, 50], "color": "green"}
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": predicted_mpg
                        }
                    }
                ))
            st.plotly_chart(fig)
            
            # Add context about the prediction
            if predicted_mpg > 30:
                st.info("This is an excellent fuel efficiency rating!")
            elif predicted_mpg > 20:
                st.info("This is a good fuel efficiency rating.")
            else:
                st.info("This vehicle has below average fuel efficiency.")
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to prediction service: {str(e)}")
            st.warning(f"Make sure the backend service is running at {BACKEND_URL}")

# Add information about the app
st.sidebar.header("about")
st.sidebar.write("""
Uses a random forest regressor to predict car mpg

""")

st.sidebar.header("Feature Impact on MPG")
feature_importance = pd.DataFrame({
    'Feature': ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin'],
    'Importance': [0.12, 0.18, 0.15, 0.25, 0.05, 0.15, 0.10]  # Example values
})

st.sidebar.bar_chart(feature_importance.set_index('Feature'))
