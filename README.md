# geospatial-sentiment Dashboard.-svm-model
# Project Title: Geospatial Sentiment Analysis Dashboard
# Description:
This project presents a comprehensive analysis tool, the "Geospatial Sentiment Analysis Dashboard," designed for monitoring and visualizing customer sentiments from reviews across various geographical locations. The dashboard leverages sentiment analysis algorithms to process textual data from customer reviews, categorizing them into positive or negative sentiments. It then visualizes this data on a global scale, enabling businesses to gain insights into customer satisfaction and opinions based on different regions and time frames.

# Features:
Sentiment Prediction: Utilizes natural language processing (NLP) techniques to predict sentiments from user-submitted text or bulk CSV file uploads. The model processes and classifies the sentiment as either positive or negative based on the textual content of the reviews.
Data Visualization: Implements interactive visualizations such as global sentiment distribution maps, sentiment trends over time, and detailed country-specific sentiment analysis. The dashboard uses Plotly for dynamic charts and graphs, and matplotlib for generating word clouds, enhancing user interaction and data interpretability.
Geospatial Analysis: Links sentiment data to geographical locations using the pycountry library, allowing the visualization of sentiment distribution across different countries. This feature helps identify regions with high customer satisfaction or areas that may require improved service or products.
Advanced Analytics: Provides detailed analytics like sentiment percentages, comparison of actual and predicted sentiments, and a classification report that includes precision, recall, and F1 scores. It also features a confusion matrix for an in-depth analysis of the model's performance.
Interactive User Interface: Built using Streamlit, the dashboard offers a user-friendly interface with sliders, buttons, and dropdowns for easy navigation and interaction. Users can filter data based on specific years, months, or custom text searches.
# Technologies Used:
Python: Core programming language.
Streamlit: For creating the web-based interactive dashboard.
Pandas & NumPy: For data manipulation and calculations.
NLTK: For text preprocessing, tokenization, and lemmatization.
Scikit-learn: For model training and sentiment classification.
Joblib: For model serialization and deserialization.
Plotly: For interactive charts.
WordCloud and Matplotlib: For generating and displaying word clouds.
Pycountry: For converting country codes to country names.
# Setup and Installation:
Clone the repository to your local machine.
Ensure Python 3.8+ is installed.
Install all required dependencies using pip install -r requirements.txt.
Run the dashboard using the command streamlit run app.py.
# Data:
The project uses a dataset containing reviews labeled with sentiments and associated store locations in country code format. This dataset allows the application to map sentiments to specific countries, providing a global view of sentiment analysis results.

# Usage:
The dashboard is ideal for businesses seeking to understand customer sentiments across different markets, for researchers studying sentiment trends globally, or for data analysts presenting insights into customer feedback.

This project is not only a powerful tool for visual sentiment analysis but also serves as a practical example of applying modern NLP techniques in real-world applications.
