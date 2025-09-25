import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

dados = pd.read_csv('taxa_selic.csv', sep=';', decimal=',')
#print(f"{len(dados)}\n")

dados["Data"] = pd.to_datetime(dados["Data"], dayfirst=True)
dados.sort_values(by="Data", inplace=True)

dados_pre = dados[dados["Data"] < "2020-03-01"].copy()
dados_pandemia = dados[(dados["Data"] >= "2020-03-01") & (dados["Data"] <= "2021-12-31")].copy()
dados_pos = dados[(dados["Data"] > "2021-12-31") & (dados["Data"] <= "2022-12-31")].copy()


#Pré Pandemia

dados_pre["Dias"] = (dados_pre["Data"] - dados_pre["Data"].min()).dt.days

x_pre = dados_pre["Dias"].values.reshape(-1, 1)
y_pre = dados_pre["Taxa (% a.a.)"].values

modelo_pre = LinearRegression()
modelo_pre.fit(x_pre, y_pre)

y_pre_pred = modelo_pre.predict(x_pre)


#Pandemia

dados_pandemia["Dias"] = (dados_pandemia["Data"] - dados_pandemia['Data'].min()).dt.days

x_pandemia = dados_pandemia['Dias'].values.reshape(-1, 1)
y_pandemia = dados_pandemia["Taxa (% a.a.)"].values

modelo_pandemia = LinearRegression()
modelo_pandemia.fit(x_pandemia, y_pandemia)

y_pandemia_pred = modelo_pandemia.predict(x_pandemia)


#Pós Pandemia

dados_pos['Dias'] = (dados_pos['Data'] - dados_pos['Data'].min()).dt.days

x_pos = dados_pos["Dias"].values.reshape(-1, 1)
y_pos = dados_pos["Taxa (% a.a.)"].values

modelo_pos = LinearRegression()
modelo_pos.fit(x_pos, y_pos)

y_pos_pred = modelo_pos.predict(x_pos)


print(f"""
                Dados Obtidos
      
Coeficiente angular pré-pandemia: {modelo_pre.coef_[0]}
Intercepto pré-pandemia: {modelo_pre.intercept_}
      
Coeficiente angular durante a pandemia: {modelo_pandemia.coef_[0]}
Intercepto durante a pandemia: {modelo_pandemia.intercept_}

Coeficiente angular pós-pandemia: {modelo_pos.coef_[0]}
Intercepto pós-pandemia: {modelo_pos.intercept_}\n

""")


# Gráfico

plt.figure(figsize=(12,6))

plt.scatter(dados_pre["Data"], y_pre, color = 'blue', label = 'Real (Pré)', s = 10)
plt.plot(dados_pre["Data"], y_pre_pred, color = 'royalblue', label = 'Regressão Linear (Pré)')

plt.scatter(dados_pandemia["Data"], y_pandemia, color = 'green', label = 'Real (Pandemia)', s = 10)
plt.plot(dados_pandemia["Data"], y_pandemia_pred, color = 'lightgreen', label = 'Regressão Linear (Pandemia)')

plt.scatter(dados_pos["Data"], y_pos, color = 'orange', label = 'Real (Pós)', s = 10)
plt.plot(dados_pos["Data"], y_pos_pred, color = 'salmon', label = 'Regressão Linear (Pós)')

plt.title('Comparação da Taxa Selic por Período')
plt.xlabel('Dias desde o início de cada período')
plt.ylabel('Taxa selic (% a.a.)')
plt.legend()
plt.grid(True)
plt.show()