import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Gerçek veri seti — 442 diyabet hastası, 10 değişken
veri = load_diabetes()
X = pd.DataFrame(veri.data, columns=veri.feature_names)
y = pd.Series(veri.target, name="Hastalık İlerlemesi")

# %80 eğitim, %20 test
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2)

# Model kur ve eğit
model = LinearRegression()
model.fit(X_egitim, y_egitim)

# Tahmin yap
y_tahmin = model.predict(X_test)

# Metrikler
r2   = r2_score(y_test, y_tahmin)
mse  = mean_squared_error(y_test, y_tahmin)
rmse = np.sqrt(mse)

print("=" * 52)
print("📌 ÇOKLU DOĞRUSAL REGRESYON")
print("📋 Veri: Diyabet Veri Seti (sklearn)")
print("=" * 52)
print(f"Toplam Veri Sayısı        : {X.shape[0]}")
print(f"Değişken Sayısı           : {X.shape[1]}")
print(f"Eğitim Seti               : {len(X_egitim)} satır")
print(f"Test Seti                 : {len(X_test)} satır")
print("-" * 52)
print("Katsayılar (Coefficients):")
for col, katsayi in zip(X.columns, model.coef_):
    print(f"  {col:<20} : {katsayi:8.2f}")
print(f"  {'Sabit (Intercept)':<20} : {model.intercept_:8.2f}")
print("-" * 52)
print(f"R² Skoru                  : {r2:.4f}")
print(f"MSE (Ort. Kare Hata)      : {mse:.4f}")
print(f"RMSE (Kök Ort. Kare Hata) : {rmse:.4f}")
print("-" * 52)
print("Tahmin vs Gerçek (ilk 5):")
karsilastirma = pd.DataFrame({
    "Gerçek"  : y_test.values[:5].round(1),
    "Tahmin"  : y_tahmin[:5].round(1),
    "Fark"    : abs(y_test.values[:5] - y_tahmin[:5]).round(1)
})
print(karsilastirma.to_string(index=False))
print("-" * 52)
if r2 > 0.7:
    print("✅ Model veriyi iyi açıklıyor! (R² > 0.70)")
elif r2 > 0.4:
    print("⚠️  Model orta düzeyde açıklıyor. (R² > 0.40)")
else:
    print("❌ Model veriyi zayıf açıklıyor. (R² < 0.40)")
print("=" * 52)
