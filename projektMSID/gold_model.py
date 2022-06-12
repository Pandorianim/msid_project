def model_fun(x, a, b, c, d, e, f):
    return a * x['kills'] + b * x['deaths'] + c * x['assists'] + d * x['minions'] + e * x['duration'] + f


class GoldModel:
    def __init__(self, params):
        self.params = params

    def run(self, x):
        return model_fun(x, *self.params)
