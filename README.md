# 📈 Project: Stock Price Prediction Using PySpark

A scalable financial analytics pipeline built with **PySpark** and **Matplotlib**. This project predicts stock closing prices using real historical stock market data using a distributed machine learning workflow.

---

## 🏛️ Project Philosophy
This project is designed as an **end-to-end data engineering and machine learning pipeline**. It demonstrates how raw stock market data can be transformed into meaningful predictive insights and attractive visual analytics. By leveraging **Apache Spark**, the project is built for scalability and can be extended from a single stock dataset to much larger financial datasets in real-world environments.

## 🚀 Key Technical Features
- **Scalable Ingestion:** Distributed CSV loading using PySpark DataFrames
- **Data Cleaning:** Mean-based missing value handling for numerical stock features
- **Feature Engineering:** Creation of lag-based features and moving average features for historical trend capture
- **Realistic Forecasting Setup:** Time-based train-test split to preserve chronological order
- **Regression Modeling:** **Linear Regression** model implemented with PySpark MLlib
- **Attractive Visual Output:** Three separate plots for actual trend, predicted trend, and actual vs predicted comparison

## 📊 Model Performance
The model is evaluated on a chronological 20% test split using regression metrics:
- **RMSE:** Measures average prediction error
- **R² Score:** Measures goodness of fit

The model demonstrates good predictive alignment with historical stock price trends on the selected dataset.

## 📂 Repository Structure
```text
Stock-Price-Prediction-Using-PySpark/
├── code/
│   ├── stock_price_prediction.py
│   └── requirements.txt
├── dataset/
│   └── stock_data.csv
├── output/
│   ├── actual_stock_price.png
│   ├── predicted_stock_price.png
│   ├── actual_vs_predicted_overlay.png
│   └── Measures.png
├── ppt/
│   └── Stock_Price_Prediction_Presentation.pptx
├── report/
│   └── Stock_Price_Prediction_Report.pdf
└── README.md
