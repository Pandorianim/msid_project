from scipy.optimize import curve_fit

import gold_model
from sklearn import metrics
import pandas as pd
from riotwatcher import LolWatcher
import time
from main import API_KEY


def download_data():
    try:
        learning = pd.read_pickle("data.pkl")
        return learning
    except Exception:
        watcher = LolWatcher(API_KEY)
        my_region = "eun1"
        me = watcher.summoner.by_name(my_region, "pandorianim")

        matches = watcher.match.matchlist_by_puuid(
            my_region, me["puuid"], 0, 100, 900
        )  # 400 draft 900 arurf 430 blind
        print(matches)
        print(len(matches))
        i = 0
        output = []
        try:
            for match in matches:
                if i == 70:
                    time.sleep(120)
                match_detail = watcher.match.by_id(my_region, match)
                characters = match_detail["info"]["participants"]
                for participant in characters:
                    if participant["puuid"] == me["puuid"]:
                        i += 1
                        participant["gameDuration"] = match_detail["info"]["gameDuration"]
                        output.append(participant)
                        print(i)
            df = pd.DataFrame(output)
        except Exception:
            pass

        kills = df["kills"].values
        deaths = df["deaths"].values
        assists = df["assists"].values
        minions = df["totalMinionsKilled"].values
        game_duration = df["gameDuration"].values
        gold = df["goldEarned"].values
        damage = df["totalDamageDealtToChampions"].values

        good_frame = pd.DataFrame(
            list(zip(kills, deaths, assists, minions, game_duration, gold, damage)),
            columns=[
                "kills",
                "deaths",
                "assists",
                "minions",
                "duration",
                "gold",
                "damage",
            ],
        )

        good_frame["duration"] /= 60

        print(len(good_frame))

        date_frame = good_frame.copy()
        date_frame.to_pickle("./data.pkl")

        download_data()


def get_min_max_of_frame(frame):
    min_col = {}
    max_col = {}
    for col in frame:
        max_col[col] = frame[col].max()
        min_col[col] = frame[col].min()

    result = pd.DataFrame([min_col, max_col], index=['min', 'max'])
    return result


def get_x_and_y(learn, test, y_type):
    learnX = learn[["kills", "deaths", "assists", "minions", "duration"]].copy()
    learnY = learn[[y_type]].copy()
    testX = test[["kills", "deaths", "assists", "minions", "duration"]].copy()
    testY = test[[y_type]].copy()

    return learnX, learnY, testX, testY


def filter_data(data, minDuration):
    data = data[data.duration >= minDuration]
    return data

def chose_cut_point(data, start_index, end_index):
    result = []
    data = data
    for i in range(start_index, end_index):
        learn = data[:i]
        test = data[i:]
        test = test.reset_index()
        del test["index"]
        l_x_g, l_y_g, t_x_g, t_y_g = get_x_and_y(learn, test, "gold")

        paramsGold, _ = curve_fit(
            gold_model.model_fun, xdata=l_x_g, ydata=l_y_g.values.ravel()
        )
        modelGold = gold_model.GoldModel(paramsGold)
        y_pred_g = modelGold.run(t_x_g)
        blad = round(metrics.mean_absolute_error(t_y_g, y_pred_g))
        result.append((blad, i))
    result.sort(key=lambda tup: tup[0])
    return result[0][1]
