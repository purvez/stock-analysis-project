# 📊 Stock Market Trend Analysis

A Python project for analyzing stock market trends with interactive visualization using **Streamlit**.  

The project includes:

- **Simple Moving Average (SMA)** calculation with sliding window optimization  
- **Upward and downward streaks** detection  
- **Daily returns** computation  
- **Maximum profit calculation** (Best Time to Buy and Sell Stock II)  
- **Visualization** of closing prices, SMA, and streaks with Plotly  

---

## 🔹 Project Structure

```
stock_trend_analysis/
│── app.py                 # Main Streamlit app
│── analysis.py            # Core analysis functions (SMA, returns, runs, max profit)
│── visualization.py       # Plotting functions (Plotly)
│── validation.py          # Test cases for validation
│── data/                  # Folder for CSV datasets
│── requirements.txt       # Python dependencies
```

---

## ⚙️ Requirements

- Python 3.8+  
- Libraries: `streamlit`, `pandas`, `matplotlib`, `seaborn`, `plotly`  

Install via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 1. Set up a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows (cmd):**
  ```cmd
  .venv\Scripts\activate
```
- **Windows (PowerShell):**
  ```powershell
.\.venv\Scripts\Activate.ps1
```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default web browser at:

```
http://localhost:8501
```

---

### 4. Upload Stock Data

The app accepts **CSV files** with at least the following columns:

- `Date` (YYYY-MM-DD format recommended)  
- `Open`  
- `High`  
- `Low`  
- `Close`  
- `Volume`  

The app will calculate SMA, daily returns, upward/downward runs, and max profit, and visualize the results.

---

## 🧩 Development Notes

- **SMA calculation** uses a sliding window approach for O(n) efficiency  
- **Upward/downward runs** track consecutive days of price increase or decrease  
- **Max profit calculation** follows LeetCode “Best Time to Buy and Sell Stock II” problem  
- **Validation** includes test cases against Pandas `.rolling().mean()` and manual calculations  

---

## 🔄 Optional: Stop the App

Press **CTRL + C** in the terminal where the app is running to stop the Streamlit server.

---

## 💡 Tips

- You can extend `analysis.py` for custom indicators  
- `visualization.py` uses Plotly, but you can switch to Matplotlib or Seaborn if desired  
- The `data/` folder is optional; you can load CSVs from any path in the app  

---

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)  
- [Plotly Documentation](https://plotly.com/python/)  
- [LeetCode Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)
