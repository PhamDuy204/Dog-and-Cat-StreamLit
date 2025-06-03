from sklearn.linear_model import LogisticRegression
from data_preparation import get_data
from preprocessing import *
import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
class Model:
    def __init__(self,**kwarg):
        self.model =LogisticRegression(**kwarg,n_jobs=-1)
    def train(self):
        if not os.path.exists('./checkpoints/best_model.pkl'):
            fearues,label=get_data()
            print(fearues.shape)
            self.model.fit(fearues,label)
            with open('./checkpoints/best_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        else:pass
    def infer(self,feature):
        if os.path.exists('./checkpoints/best_model.pkl'):
            with open(
                './checkpoints/best_model.pkl','rb'
            ) as f:
                self.model = pickle.load(f)
        batch = np.array([feature])
        pred= self.model.predict(batch)[0]
        score = self.model.predict_proba(batch)[0,pred]
        return 'Dog' if pred == 0 else 'Cat',score
