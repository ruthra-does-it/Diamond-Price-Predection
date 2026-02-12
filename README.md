# Diamond Price Prediction

This project performs comprehensive data analysis and machine learning on the diamonds dataset to predict diamond prices. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and a Streamlit web app for interactive visualization and prediction.

## Dataset

The dataset used is `diamonds.csv`, which contains information about diamonds including carat, cut, color, clarity, depth, table, price, and dimensions (x, y, z).

## Features

- **Data Preprocessing**: Handling missing values, outlier removal, skewness correction.
- **Exploratory Data Analysis**: Visualizations including distributions, correlations, and categorical analysis.
- **Feature Engineering**: Creating new features like volume, price per carat, and encoding categorical variables.
- **Machine Learning Models**:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - K-Nearest Neighbors
  - Artificial Neural Network (ANN)
- **Clustering**: K-Means clustering to segment diamonds.
- **Streamlit App**: Interactive dashboard for data exploration and price prediction.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ruthra-does-it/Diamond-Price-Predection.git
   cd Diamond-Price-Predection
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Analysis Script

Execute the main analysis script:
```
python diamond.py
```
This will perform the full data analysis, train models, and save them as `.pkl` files.

### Running the Streamlit App

Launch the interactive web app:
```
streamlit run app.py
```
Open your browser to `http://localhost:8501` to explore the data and make predictions.

## Models

- `best_price_model.pkl`: Random Forest model for price prediction.
- `best_cluster_model.pkl`: K-Means clustering model.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow
- streamlit

## Project Structure

```
Diamond-Price-Predection/
├── diamond.py          # Main analysis script
├── app.py              # Streamlit application
├── diamonds.csv        # Dataset
├── best_price_model.pkl    # Saved price prediction model
├── best_cluster_model.pkl  # Saved clustering model
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## Contributing

Feel free to fork the repository and submit pull requests for improvements.

## License

This project is open-source. Use it as you wish.