# 🚀 Quantitative Trader Intelligence Dashboard - Shahid Mulani

### *Uncovering Alpha Through Sentiment Analysis & Statistical Modeling*

<img width="1858" height="796" alt="Screenshot 2025-11-18 184907" src="https://github.com/user-attachments/assets/b67b7a23-57de-466a-9c0f-50d8fe0f50f6" />

---

## 🎯 Executive Summary

**The Challenge:** In the high-frequency world of crypto trading, does market sentiment (Fear vs. Greed) reliably predict asset performance? And can we mathematically prove a structural edge?

**The Solution:** I engineered a production-grade **Quantitative Analytics Engine** deployed as an interactive web app. Instead of static charts, this tool processes **200,000+ historical trades** to perform real-time hypothesis testing, risk modeling, and strategy validation.

**The Verdict:** The analysis uncovered a statistically significant **"Momentum Anomaly."** Contrary to the popular "contrarian" investing wisdom, this specific trading cohort maximizes Alpha during **"Extreme Greed"** while suffering capital decay during **"Extreme Fear."**

<img width="1814" height="517" alt="Screenshot 2025-11-18 184923" src="https://github.com/user-attachments/assets/5e054292-fb77-4f80-a898-33dfc9164d8e" />


---

## 📊 Strategic Findings (The "Alpha")

My algorithmic analysis revealed three critical patterns for PnL optimization:

### 1. The "Greed" Advantage (Momentum is King)
* **Observation:** Trader profitability maximizes when the Market Sentiment Index **> 75**.
* **Data Evidence:**
    * **Win Rate:** **46.5%** (Significantly higher than the 37% floor in Fear regimes).
    * **Expectancy:** **$67.89** Avg PnL per trade (Highest across all zones).
    * **Efficiency:** The Sharpe Ratio peaks here, indicating the best risk-adjusted returns.
* **Actionable Strategy:** Increase capital allocation multipliers by **50-75%** during high-sentiment periods to capture the momentum premium.

### 2. The "Fear" Trap (Capital Leak)
* **Observation:** "Extreme Fear" periods are characterized by negative expectancy and excessive volatility.
* **Data Evidence:**
    * **Volatility:** PnL Standard Deviation spikes to **$1,136**, indicating chaotic price action.
    * **Behavioral Bias:** Despite poor performance, traders deploy their **largest position sizes** (~$7,816) during Fear, attempting to "catch falling knives."
* **Actionable Strategy:** Implement a hard **"Volatility Circuit Breaker"** to cap position sizes at $2,500 when Sentiment drops below 25.

---

## 🧠 Quantitative Methodology

To ensure these insights were not random noise, I integrated rigorous mathematical models directly into the application logic:

### 🧪 1. Statistical Hypothesis Testing
* **Method:** Welch’s T-Test (Two-Sample, Unequal Variance).
* **Hypothesis:** `Null (H0): Mean PnL (Greed) <= Mean PnL (Fear)`
* **Result:** **P-Value < 0.05**.
* **Conclusion:** We reject the null hypothesis with **>95% confidence**. The performance gap is structural and statistically significant.

### 📐 2. The Kelly Criterion (Money Management)
Used to calculate the mathematically optimal bet size to maximize geometric growth.

$$f^* = W - \frac{1 - W}{R}$$

* **Application:** The dashboard dynamically suggests aggressive sizing during "Greed" and defensive sizing during "Fear" based on the live Win Rate ($W$) and Risk/Reward Ratio ($R$).

### ⚖️ 3. Sharpe Ratio (Efficiency)

$$S_p = \frac{R_p - R_f}{\sigma_p}$$

* **Application:** Used to demonstrate that while "Fear" periods have high volume, they offer the lowest efficiency per unit of risk taken.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | `Streamlit` | Built a responsive, interactive dashboard with Glassmorphism UI. |
| **Data Engine** | `Pandas` / `NumPy` | Vectorized data processing for high-performance filtering. |
| **Visualization** | `Seaborn` / `Matplotlib` | Generated comparative bar charts and trend lines. |
| **Statistics** | `SciPy` | Integrated scientific computing for real-time inference. |

---

## 🚀 Local Installation

If you wish to run the analysis engine locally:

**1. Clone the Repository**
git clone [https://github.com/your-username/trader-insights.git](https://github.com/your-username/trader-insights.git)

**Contact**
**Candidate: Shahid Mulani Role: Junior Data Scientist Applicant**

Ready to drive data-driven decisions at Prime Trade AI.

