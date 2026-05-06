import numpy as np
from scipy.optimize import curve_fit

def _fit_exponential(x_data, y_data):
    """OLS-seeded NLLS for exponential decay (additive noise)."""
    y_safe = np.clip(y_data, 1e-9, None)
    b_guess, log_a_guess = np.polyfit(x_data, np.log(y_safe), 1)
    a_guess = np.exp(log_a_guess)
    print(f"Guess: a={a_guess}, b={b_guess}")

    def model(x, a, b):
        return a * np.exp(b * x)

    # Bound amplitude to positive, decay rate to <= 0
    params, _ = curve_fit(
        model,
        x_data,
        y_data,
        p0=[a_guess, b_guess],
        bounds=([0, -np.inf], [np.inf, 0.0]),
        maxfev=10000
    )
    return model(x_data, *params)

x = np.arange(10)
y = np.ones(10) + np.random.normal(0, 0.1, 10) # Noisy flat data
print("Data:", y)
_fit_exponential(x, y)
