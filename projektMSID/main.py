from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn import metrics


import seaborn as sns
import damage_model
import gold_model
import numpy as np
import utilities as ut

#stałe parametryczne]
API_KEY = "RGAPI-a95e3ab3-bb38-46b3-8f6a-9fa9b1d8df9e"
MINTEST = 16
MINTRAINING = 60
MINGAMELENGTH = 11


if __name__ == "__main__":
    data = ut.download_data()
    # filtrujemy z danych za krotkie mecze (ekstremalny przypadek, potencjalne dzielenie przez 0)
    data = ut.filter_data(data, MINGAMELENGTH)

    #wartosci do zakresow
    # print(ut.get_min_max_of_frame(data))

    print("Wskutek filtrowania zbyt krótkich meczów usunięto: ", 100-len(data), "meczów")
    data = data.reset_index()
    del data["index"]

    cut_point = ut.chose_cut_point(data, MINTRAINING, len(data)-MINTEST)
    print("Data została podzielona na indexie: ", cut_point)

    learn = data[:cut_point]
    test = data[cut_point:]

    test = test.reset_index()
    del test["index"]
    test = test.reset_index()

    l_x_g, l_y_g, t_x_g, t_y_g = ut.get_x_and_y(learn, test, "gold")

    paramsGold, _ = curve_fit(
        gold_model.model_fun, xdata=l_x_g, ydata=l_y_g.values.ravel()
    )
    print("Parametry do funkcji: ",paramsGold)
    modelGold = gold_model.GoldModel(paramsGold)
    y_pred_g = modelGold.run(t_x_g)

    print("Średni błąd absolutny modelu przewidującego zdobyte złoto:")
    print(round(metrics.mean_absolute_error(t_y_g, y_pred_g), 2))
    print("Średni błąd (w procentach) modelu przewidującego zdobyte złoto:")
    print(round(metrics.mean_absolute_percentage_error(t_y_g, y_pred_g)*100, 2), "%")

    listaPred = list(y_pred_g)
    listaTrue = list(test["gold"])
    listaIndex = list(test['index'])

    # Wykres przedstawiający doposwanie modelu do prawdziwych wartosci
    X_axis = np.linspace(
        start=test["duration"].min(), stop=test["duration"].max(), num=len(test)
    )
    sns.scatterplot(x="index", y="gold", data=test)
    sns.scatterplot(x=listaIndex, y=listaPred, color="red", label='Pred')
    plt.show()
    l_x_d, l_y_d, t_x_d, t_y_d = ut.get_x_and_y(learn, test, "damage")

    paramsDamage, _ = curve_fit(
        damage_model.model_fun, xdata=l_x_d, ydata=l_y_d.values.ravel()
    )
    modelDamage = damage_model.DamageModel(paramsDamage)
    y_pred_d = modelDamage.run(t_x_g)

    print("Średni błąd absolutny modelu przewidującego zadane obrażenia:")
    print(round(metrics.mean_absolute_error(t_y_d, y_pred_d), 2))
    print("Średni błąd (w procentach) modelu przewidującego zadane obrażenia:")
    print(round(metrics.mean_absolute_percentage_error(t_y_d, y_pred_d)*100, 2), "%")

#do sprawozdania program overleaf