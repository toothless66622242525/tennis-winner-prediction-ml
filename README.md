# tennis-winner-prediction-ml

## Project Overview

This project is an end-to-end machine learning pipeline designed to predict the outcomes of ATP tennis matches. The entire data science lifecycle is covered, from raw data ingestion to a final, real-world prediction, demonstrating a complete and reproducible workflow.

The key goal is to systematically evaluate and compare different classification models to find the most accurate predictor based on the available data, and then apply it to a real match.

## Technologies Used

*   **Python:** The core language for the entire pipeline.
*   **Pandas:** For data manipulation, cleaning, and feature engineering.
*   **Scikit-learn:** For data preprocessing (`OneHotEncoder`, `ColumnTransformer`), model training (`LogisticRegression`, `DecisionTree`), and evaluation (`accuracy_score`).
*   **XGBoost:** To implement the high-performance gradient boosting model.
*   **JupyterLab:** As the primary IDE for interactive development and analysis.

## The Pipeline

This project is structured as a modular pipeline, with each notebook representing a distinct stage:

1.  **`01_data_preparation.ipynb`**: **(Data Prep Stage)**
    *   Loads the raw `atp_tennis.csv` dataset.
    *   Performs initial filtering to work only with modern, high-quality data (post-2005).
    *   Creates the binary `Target` variable required for classification.
    *   Engineers new features (`Rank_Difference`, `Pts_Difference`).
    *   Handles missing values (`.fillna()`).
    *   **Output:** Saves the final, clean feature matrix `X` and target vector `y` into `prepared_X.pkl` and `prepared_y.pkl` files for the next stage.

2.  **`02_model_training.ipynb`**: **(Modeling & Evaluation Stage)**
    *   Loads the prepared data from the `.pkl` files.
    *   Splits the data into training and testing sets using `train_test_split`.
    *   Builds a preprocessing pipeline using `ColumnTransformer` to handle categorical features (`Surface`).
    *   Trains, evaluates, and compares multiple models: `LogisticRegression`, `DecisionTree`, and `XGBoost`.
    *   Identifies the champion model (XGBoost) with an accuracy of **~70%**.

3.  **`03_real_world_prediction.ipynb`**: **(Application Stage)**
    *   Retrains the champion model (XGBoost) on 100% of the available data to maximize its predictive power.
    *   Takes manually gathered data for a real-world, future match (e.g., Shelton vs. Khachanov).
    *   Applies the trained preprocessing pipeline and the final model to generate a concrete prediction and probability estimate.

## How to Run

1.  **Clone the repository.**
2.  **Install dependencies:** `pip install pandas scikit-learn xgboost`
3.  **Execute the notebooks sequentially:** `01_...` -> `02_...` -> `03_...`. Each notebook uses the artifacts (output files) from the previous one.

## Data Source

The `atp_tennis.csv` file used in this project is based on the [ATP Tennis 2000-2023 (Daily Pull)](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) dataset from Kaggle, created by user **dissfya**. Huge thanks to them for their incredible work in providing clean, structured data to the community.

The version included in this repository is current as of **July 31, 2025**.

To work with the very latest match data, you can download the updated version directly from the Kaggle dataset page and replace the existing `atp_tennis.csv` file before running the pipeline.

## Conclusion & Real-World Validation

The project successfully demonstrates the creation of a complete and robust ML pipeline. The final XGBoost model, after hyperparameter tuning, was identified as the champion, achieving a stable accuracy of **~71%** on the available features.

The key insight from the modeling process is the identification of a "data ceiling". While the model performs significantly better than random chance, further substantial improvements would require more nuanced features beyond basic rankings and stats, such as player's current form, head-to-head history, and fatigue metrics.

The true validation of the pipeline came from its application to a real-world match:

*   **Match:** Ben Shelton vs. Karen Khachanov (Toronto 2025 Final)
*   **Final Model's Prediction:** **Ben Shelton to win**
*   **Model's Confidence:** **57.82%**

This demonstrates the model's ability to take new, unseen data and produce a reasonable, quantifiable forecast that aligns with expert opinion (favoring the higher-ranked player but acknowledging the opponent's strength). The project proves that even a baseline model, when built correctly, can be a powerful tool for turning historical data into forward-looking insights.

