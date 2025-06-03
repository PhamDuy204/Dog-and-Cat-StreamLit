from sklearn.svm import SVC
from data_preparation import get_data
from preprocessing import *
import pickle
import os
class Model:
    def __init__(self,**kwarg):
        self.model =SVC(**kwarg,probability=True)
    def train(self):
        if not os.path.exists('/home/duy/mlops/Demo_Streamlit/checkpoints/best_model.pkl'):
            fearues,label=get_data()
            print(fearues.shape)
            self.model.fit(fearues,label)
            with open('/home/duy/mlops/Demo_Streamlit/checkpoints/best_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        else:pass
    def infer(self,feature):
        if os.path.exists('/home/duy/mlops/Demo_Streamlit/checkpoints/best_model.pkl'):
            with open(
                '/home/duy/mlops/Demo_Streamlit/checkpoints/best_model.pkl','rb'
            ) as f:
                self.model = pickle.load(f)
        batch = np.array([feature])
        pred= self.model.predict(batch)[0]
        score = self.model.predict_proba(batch)[0,pred]
        return 'Dog' if pred == 0 else 'Cat',score
