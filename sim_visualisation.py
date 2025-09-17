import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, lognorm
import os

# Ensure the 'figures' directory exists
os.makedirs("figures", exist_ok=True)

# ------------------------
# Parameters
# ------------------------
N = 10000
years = 5
discount_rate = 0.08

deal_value = 500  # $M (example)
combined_revenue = 120  # $M (example)
announced_cost_synergies = 10  # $M annual run-rate
announced_integration = 12  # $M

rev_mu, rev_sigma = 0.01, 0.005
cost_mu, cost_sigma = announced_cost_synergies, 3
int_median, int_sigma_ln = announced_integration, 0.6

rev_ramp = np.array([0.2, 0.5, 0.8, 0.95, 1.0])
cost_ramp = np.array([0.5, 0.9, 1.0, 1.0, 1.0])
rho = 0.4

# ------------------------
# Sampling helpers
# ------------------------
def sample_truncnorm(mu, sigma, size):
    a, b = (0 - mu) / sigma, np.inf
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)

def sample_lognormal(median, sigma_ln, size):
    mu_ln = np.log(median)
    return lognorm.rvs(sigma_ln, scale=np.exp(mu_ln), size=size)

# ------------------------
# Simulation
# ------------------------
rng = np.random.default_rng(42)

cov = np.array([[1, rho], [rho, 1]])
L = np.linalg.cholesky(cov)
z = rng.standard_normal((2, N))
corr_draws = L @ z

rev_pct_draw = rev_mu + rev_sigma * corr_draws[0]
rev_pct_draw = np.clip(rev_pct_draw, 0, None)

cost_draw = rng.normal(cost_mu, cost_sigma, N)
cost_draw = np.clip(cost_draw, 0, None)

int_draw = sample_lognormal(int_median, int_sigma_ln, N)
int_draw = int_draw * (1 + 0.5 * corr_draws[1])

npvs = []
for i in range(N):
    annual_rev = rev_pct_draw[i] * combined_revenue
    annual_cost = cost_draw[i]
    synergies = (annual_rev * rev_ramp) + (annual_cost * cost_ramp)
    discounted = np.sum(synergies / (1 + discount_rate) ** np.arange(1, years + 1))
    npv = discounted - int_draw[i]
    npvs.append(npv)

df = pd.DataFrame({
    "rev_pct": rev_pct_draw,
    "cost_synergy": cost_draw,
    "integration_cost": int_draw,
    "npv": npvs
})
df.to_csv("simulation_results_gtcr_fmg.csv", index=False)

# ------------------------
# Results
# ------------------------
mean_npv = df["npv"].mean()
median_npv = df["npv"].median()
p_positive = (df["npv"] > 0).mean()
var5 = df["npv"].quantile(0.05)

print("---- Results ----")
print(f"Mean NPV:     ${mean_npv:,.1f}M")
print(f"Median NPV:   ${median_npv:,.1f}M")
print(f"P(NPV>0):     {p_positive:.1%}")
print(f"VaR (5%):     ${var5:,.1f}M")

# ------------------------
# Figures
# ------------------------
# 1. Histogram
plt.figure()
plt.hist(df["npv"], bins=50, color="steelblue", edgecolor="black")
plt.axvline(mean_npv, color="red", linestyle="--", label="Mean")
plt.axvline(median_npv, color="green", linestyle="--", label="Median")
plt.title("Distribution of Synergy NPV (GTCR-FMG)")
plt.xlabel("NPV ($M)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("figures/npv_histogram_gtcr.png", dpi=300)

# 2. CDF
'''plt.figure()
sorted_npvs = np.sort(df["npv"])
cdf = np.arange(1, N + 1) / N
plt.plot(sorted_npvs, cdf, color="steelblue")
plt.axhline(p_positive, color="red", linestyle="--")
plt.title("CDF of Synergy NPV (GTCR-FMG)")
plt.xlabel("NPV ($M)")
plt.ylabel("Cumulative Probability")
plt.savefig("figures/cdf_gtcr.png", dpi=300)

# 3. Tornado / driver importance (annotated)
corrs = df.corr()["npv"].drop("npv").sort_values()
colors = ["green" if x > 0 else "red" for x in corrs]

plt.figure(figsize=(8,5))
bars = plt.barh(corrs.index, corrs.values, color=colors)
plt.title("Driver Importance (Correlation with NPV)")
plt.xlabel("Correlation with NPV")

# Annotate each bar with correlation value
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.02 if width > 0 else width - 0.05,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}",
        va='center',
        ha='left' if width > 0 else 'right',
        fontsize=10,
        color="black"
    )

# Add simple legend
plt.text(1.05, len(corrs)-0.5, "Positive impact ↑", color="green", fontsize=10)
plt.text(1.05, len(corrs)-1.2, "Negative impact ↓", color="red", fontsize=10)

plt.tight_layout()
plt.savefig("figures/tornado_gtcr_annotated.png", dpi=300)'''
plt.show()
