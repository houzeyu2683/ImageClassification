import sklearn.metrics

class Metric:

    def __init__(self, score=None, prediction=None, target=None):
        self.score = score
        self.prediction = prediction
        self.target = target
        return
    
    def getAccuracy(self):
        score = sklearn.metrics.accuracy_score(self.target, self.prediction)
        score = round(score, 3)
        return(score)

    def getAreaUnderCurve(self):
        score = sklearn.metrics.roc_auc_score(self.target, self.score)
        score = round(score, 3)
        return(score)

    def getLoss(self):
        score = sklearn.metrics.log_loss(self.target, self.score)
        score = round(score, 3)
        return(score)

    pass
