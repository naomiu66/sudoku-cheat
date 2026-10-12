import copy

class BestCheckpoint():
    def __init__(self, model):
        self.model_state = copy.deepcopy(model.state_dict())
        self.acc_score = 0
        
    def update(self, model, accuracy):
        if accuracy > self.acc_score:
            self.model_state = copy.deepcopy(model.state_dict())
            self.acc_score = accuracy