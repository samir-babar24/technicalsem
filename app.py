import streamlit as st
from joblib import load
import pandas as pd

# Load the trained model
model = load('dialect_model.joblib')

# Custom CSS for Dark Theme and Attractive Layout
st.markdown("""
    <style>
        /* Dark Theme Background and Text */
        .stApp {
            background-color: #0e1117;  /* Dark gray-black */
            color: #ffffff;  /* White text */
        }
        
        /* Header Styling */
        h1, h2, h3 {
            color: #4db8ff;  /* Bright blue for headers */
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        
        /* Input and Button Styling */
        .stTextArea textarea {
            background-color: #1f2937;  /* Darker input bg */
            color: #ffffff;
            border: 1px solid #4db8ff;
            border-radius: 8px;
            padding: 10px;
        }
        
        .stButton button {
            background-color: #4db8ff;  /* Blue button */
            color: #0e1117;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            width: 100%;
            transition: background-color 0.3s;
        }
        
        .stButton button:hover {
            background-color: #3b82f6;  /* Darker blue on hover */
        }
        
        /* Output Box Styling */
        .output-box {
            background-color: #1f2937;  /* Dark card bg */
            border: 1px solid #4db8ff;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            font-size: 16px;
        }
        
        /* Emojis and Icons */
        .emoji {
            font-size: 24px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title with Emoji
st.title("🌟 Marathi Dialect Classifier")
st.markdown("<h3>Enter a Marathi query to predict the dialect and its standard translation!</h3>", unsafe_allow_html=True)

# User Input
query = st.text_area("Your Query (e.g., 'तुला लगिन व्हायेल का?')", height=100)

# Predict Button
if st.button("🔍 Predict"):
    if query.strip() == "":
        st.error("Please enter a query!")
    else:
        # Prepare input as Series (model expects this)
        new_queries = pd.Series([query])
        
        # Run prediction
        with st.spinner("Analyzing..."):
            predictions = model.predict(new_queries)
        
        # Display Results in a Card-like Box
        dialect = predictions[0][0]
        real_answer = predictions[0][1]
        
        st.markdown(f"""
            <div class="output-box">
                <p><strong>Predicted Dialect:</strong> {dialect} <span class="emoji">🗣️</span></p>
                <p><strong>Standard Translation (Real Answer):</strong> {real_answer} <span class="emoji">📜</span></p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<p style='text-align: center; color: #9ca3af;'>Powered by xAI & Streamlit | Data from final dataset.csv</p>", unsafe_allow_html=True)