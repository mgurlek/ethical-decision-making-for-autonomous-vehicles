import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv('excel_donusum_pozitif_negatif_final.csv', encoding='utf-8-sig')


categorical_cols = [
    'varlık', 'sağlık', 'yaş', 'ünlü', 'bağlılık', 'iyilik', 'zenginlik',
    'cinsiyet', 'uyruk', 'kilo', 'hamile', 'araç'
]


df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

X = df_encoded.drop(columns=['tercih_edildi', 'soru'])
y = df_encoded['tercih_edildi']

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X, y)

coefficients = pd.Series(log_model.coef_[0], index=X.columns)

coeffs = coefficients.to_dict()

print("=== EN YÜKSEK VE EN DÜŞÜK ETKİLİ ÖZELLİKLER ===")
print(coefficients.sort_values(ascending=False).head(15))
print(coefficients.sort_values(ascending=True).head(15))

def puan(grup, coeffs, reference_columns):
    score = 0
    if "kişi_sayısı" in grup:
        score += coeffs.get("kişi_sayısı", 0) * grup["kişi_sayısı"]

    for feature, value in grup.items():
        if feature != "kişi_sayısı":
            col = f"{feature}_{value}"
            if col in reference_columns:
                score += coeffs.get(col, 0)
    return score


# ÖRNEK SENARYOLAR
kadın_özellikleri = {
    'varlık': 'insan',
    'sağlık': 'sağlıklı',
    'kişi_sayısı': 3,
    'yaş': 'yetişkin',
    'ünlü': 'hayır',
    'bağlılık': 'sıradan',
    'iyilik': 'kötü',
    'zenginlik': 'zengin',
    'cinsiyet': 'kadın',
    'uyruk': 'türk',
    'kilo': 'zayıf',
    'hamile': 'evet',
    'araç': 'yaya'
}

kedi_özellikleri = {
    'varlık': 'insan',
    'sağlık': 'sağlıklı',
    'kişi_sayısı': 3,
    'yaş': 'yetişkin',
    'ünlü': 'hayır',
    'bağlılık': 'sıradan değil',
    'iyilik': 'iyi',
    'zenginlik': 'zengin',
    'cinsiyet': 'erkek',
    'uyruk': 'türk',
    'kilo': 'zayıf',
    'hamile': 'hayır',
    'araç': 'yaya'
    
}

# Skorları hesapla
s1 = puan(kadın_özellikleri, coeffs, X.columns)
s2 = puan(kedi_özellikleri, coeffs, X.columns)


print("\n=== KARŞILAŞTIRMA ===")
print(f"Sağlıklı Kadın Puanı: {s1:.4f}")
print(f"3 Kedi Puanı: {s2:.4f}")
print("Tercih:", "Kadın" if s1 > s2 else "Kediler")