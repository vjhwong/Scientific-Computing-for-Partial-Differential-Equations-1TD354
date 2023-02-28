import pandas as pd

# Fastest Direct method: LU
# Fastest iterative method: CGD
result = {
    "Method": ["Gauss", "LU", "Jacobi", "Conjugate Gradient Descent"],
    "Time [s.]": [4.43e-01, 3.45e-02, 7.37e-01, 1.03e-01],
    "Total iterations": ["N/A", "N/A", 2113, 1200],
}

results_df = pd.DataFrame(result)
print(results_df)
