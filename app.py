import uvicorn
from fastapi import FastAPI
import joblib
import pickle
from Diamond_Predictor import Diamond_Predictor

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
app = FastAPI()
model = joblib.load('trained_model.joblib')


@app.get('/')
def index():
    return {'message': 'Diamond Price Recommender ML API'}


@app.post('/price/predict')
def predict_diamond_type(data: Diamond_Predictor):
    data=data.dict()
    carat=data['carat']
    cut= label_encoders['cut'].transform([data['cut']])[0]
    color=label_encoders['color'].transform([data['color']])[0]
    clarity=label_encoders['clarity'].transform([data['clarity']])[0]
    depth=data['depth']
    table=data['table']
    x=data['x']
    y=data['y']
    z=data['z']
    prediction = model.predict([[carat,cut,color,clarity,depth,table,x,y,z]])

    return {
        'prediction': prediction.tolist()
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

