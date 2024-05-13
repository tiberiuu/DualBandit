import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Load the data from "data.csv"
data_path = "data.csv"
df = pd.read_csv(data_path)

# Define the model fitting and evaluation function
def fit_and_evaluate(data):
    # Prepare input and output variables
    X = data[['Tau']]
    y_knowledge = data['Knowledge']
    y_exploration_regret = data['ExplorationRegret']

    # Fit third-degree polynomial regression models for Knowledge
    poly_knowledge = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    poly_knowledge.fit(X, y_knowledge)

    # Fit third-degree polynomial regression models for ExplorationRegret
    poly_exploration_regret = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    poly_exploration_regret.fit(X, y_exploration_regret)

    # Find the Tau value at the best performance
    best_performance = data[data['Performance'] == data['Performance'].max()]

    # Calculate second derivatives at the best performance point
    tau = best_performance['Tau'].values[0]
    knowledge_coef = poly_knowledge.named_steps['linearregression'].coef_
    exploration_coef = poly_exploration_regret.named_steps['linearregression'].coef_
    # The second derivative for a 3rd degree polynomial ax^3 + bx^2 + cx + d is 6ax + 2b
    sec_deriv_knowledge = 6 * knowledge_coef[1] * tau + 2 * knowledge_coef[0]
    sec_deriv_exploration_regret = 6 * exploration_coef[1] * tau + 2 * exploration_coef[0]

    return tau, sec_deriv_knowledge, sec_deriv_exploration_regret

# Loop over each unique turbulence level and process separately
results = []
for turbulence_level in df['Turbulence'].unique():
    subset = df[df['Turbulence'] == turbulence_level]
    tau, sec_deriv_knowledge, sec_deriv_exploration_regret = fit_and_evaluate(subset)
    results.append({
        'Turbulence': turbulence_level,
        'Tau': tau,
        'Second Derivative (Knowledge)': sec_deriv_knowledge,
        'Second Derivative (ExplorationRegret)': sec_deriv_exploration_regret
    })

# Print the results
for result in results:
    print(f"Turbulence: {result['Turbulence']}, Tau: {result['Tau']}")
    print(f"Second Derivative (Knowledge): {result['Second Derivative (Knowledge)']}")
    print(f"Second Derivative (ExplorationRegret): {result['Second Derivative (ExplorationRegret)']}")
    print("-" * 30)
