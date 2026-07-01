## 📈 Multiple Linear Regression (Real Dataset)

This project applies Multiple Linear Regression to a real-world
medical dataset — the **Diabetes Dataset** from scikit-learn,
which contains data from 442 diabetes patients with 10 biological
variables (age, sex, BMI, blood pressure, and 6 serum measurements)
to predict disease progression one year after baseline.

**What I did in the code:**
1. Loaded a real dataset from sklearn (442 patients, 10 features)
2. Split the data into 80% training and 20% test sets
3. Built and trained a Linear Regression model
4. Evaluated the model using R², MSE, and RMSE metrics
5. Compared predicted vs actual values ✅

**Key Metrics:**
- **R²** → How well the model explains the data (0 to 1)
- **MSE** → Mean Squared Error (average of squared errors)
- **RMSE** → Square root of MSE, easier to interpret

**What the 50-run test showed:**
- ✅ Good  (R² > 0.70) →  0 / 50 — %0
- ⚠️ Fair  (R² > 0.40) → 41 / 50 — %82
- ❌ Weak  (R² < 0.40) →  9 / 50 — %18

The model consistently performed at a **fair level**, which is
completely expected for real biological data. Disease progression
is a complex process that a simple linear model can only partially
explain. More advanced models like **Random Forest** or **XGBoost**
would likely improve these results significantly.

**Libraries used:** `numpy`, `pandas`, `scikit-learn`
