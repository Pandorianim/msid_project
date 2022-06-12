def model_fun(x, a, b, c, d, e):
    return a * x["kills"] + b * x["deaths"] + c * x["assists"] + d * x["duration"] + e


class DamageModel:
    def __init__(self, params):
        self.params = params

    def run(self, x):
        return model_fun(x, *self.params)
