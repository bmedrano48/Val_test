# Save as app.py and run with `streamlit run app.py`

import streamlit as st
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# PERT distribution function
def pert_sample(low, mode, high, size=1, lamb=4):
    alpha = 1 + lamb * (mode - low) / (high - low)
    beta_param = 1 + lamb * (high - mode) / (high - low)
    return beta.rvs(alpha, beta_param, size=size) * (high - low) + low

# Sidebar Inputs
st.sidebar.header("Model Parameters")

starting_arr = st.sidebar.number_input("Starting ARR ($)", value=5_000_000, step=100_000)
revenue_per_customer = st.sidebar.number_input("Revenue per Customer ($)", value=5000, step=500)
inflation = st.sidebar.slider("Annual Inflation Rate (%)", 0.0, 10.0, 2.0) / 100

st.sidebar.markdown("### Growth Assumptions (%)")
growth_low = st.sidebar.slider("Low Growth", 0.0, 100.0, 15.0) / 100
growth_mode = st.sidebar.slider("Base Growth", 0.0, 100.0, 30.0) / 100
growth_high = st.sidebar.slider("High Growth", 0.0, 100.0, 70.0) / 100

st.sidebar.markdown("### Exit Multiple Assumptions")
exit_low = st.sidebar.slider("Low Exit Multiple", 1.0, 20.0, 5.0)
exit_mode = st.sidebar.slider("Base Exit Multiple", 1.0, 20.0, 7.0)
exit_high = st.sidebar.slider("High Exit Multiple", 1.0, 20.0, 12.0)

n_simulations = st.sidebar.number_input("Number of Simulations", value=10000, step=1000)

st.sidebar.markdown("### Correlation Settings")
correlation_strength = st.sidebar.slider("Y2 Growth Correlation to Y1", 0.0, 1.0, 0.6)
multiple_boost_factor = st.sidebar.slider("Multiple Boost per Growth", 0.0, 2.0, 0.5)

# Simulation
exit_values = []
revenue_per_customer *= (1 + inflation)**2

for _ in range(n_simulations):
    growth_y1 = pert_sample(growth_low, growth_mode, growth_high)[0]
    growth_y2 = growth_mode + correlation_strength * (growth_y1 - growth_mode)
    growth_y2 = np.clip(growth_y2, growth_low, growth_high)
    growth_factor = (1 + growth_y1) * (1 + growth_y2)
    future_arr = starting_arr * growth_factor

    avg_growth = (growth_y1 + growth_y2) / 2
    base_multiple = pert_sample(exit_low, exit_mode, exit_high)[0]
    growth_boost = 1 + multiple_boost_factor * ((avg_growth - growth_mode) / growth_mode)
    growth_boost = np.clip(growth_boost, 0.9, 1.5)
    final_multiple = base_multiple * growth_boost

    exit_value = future_arr * final_multiple
    exit_values.append(exit_value)

exit_values = np.array(exit_values)

# Results Display
st.title("🏁 Monte Carlo Exit Value Estimator")

st.subheader("Summary Statistics")
st.write(f"**Mean Exit Value**: ${exit_values.mean():,.0f}")
st.write(f"**Median Exit Value**: ${np.median(exit_values):,.0f}")
st.write(f"**5th Percentile**: ${np.percentile(exit_values, 5):,.0f}")
st.write(f"**95th Percentile**: ${np.percentile(exit_values, 95):,.0f}")

# Histogram
st.subheader("Distribution of Exit Values")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(exit_values / 1e6, bins=50, color="#0059b3", alpha=0.8)
ax.axvline(np.median(exit_values) / 1e6, color='red', linestyle='--', label='Median')
ax.set_xlabel("Exit Value ($MM)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)