from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal
import joblib
import pandas as pd

with open('EstateValuePredictor.pkl', 'rb') as f:
    model = joblib.load(f)

app = FastAPI()

class HousingData(BaseModel):
    latitude: Annotated[float, Field(..., description='Enter the latitude of the house', examples=[37.88])]
    longitude: Annotated[float, Field(..., description='Enter the longitude of the house', examples=[-122.23])]
    housing_median_age: Annotated[float, Field(..., gt=0, description='Enter the age of the house', examples=[52.0])]
    total_rooms: Annotated[float, Field(..., gt=0, description='Enter the total no. of rooms in that block group', examples=[750.0])]
    total_bedrooms: Annotated[float, Field(..., gt=0, description='Enter the total no. of bedrooms in that block group', examples=[190.0])]
    population: Annotated[float, Field(..., gt=0, description='Enter the population of that block group', examples=[20401.0])]
    households: Annotated[float, Field(..., gt=0, description='Enter the no. of households', examples=[117.0])]
    median_income: Annotated[float, Field(..., gt=0, description='Enter the median income', examples=[8.3014])]
    ocean_proximity: Annotated[Literal['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'] , Field(..., description='Enter the ocean proximity of the house', examples=["NEAR BAY"])]

    # because model will automatically perform these operations in pipeline
    # @computed_field
    # @property
    # def rooms_per_house(self) -> float:
    #     return self.total_rooms/self.households

    # @computed_field
    # @property
    # def bedrooms_ratio(self) -> float:
    #     return self.total_bedrooms/self.total_rooms

    # @computed_field
    # @property
    # def people_per_house(self) -> float:
    #     return self.population/self.households

@app.post('/predict')
def predict(data: HousingData):
    user_input = pd.DataFrame([{
        'latitude': data.latitude,
        'longitude': data.longitude,
        'housing_median_age': data.housing_median_age,
        'total_rooms': data.total_rooms,
        'total_bedrooms': data.total_bedrooms,
        'population': data.population,
        'households': data.households,
        'median_income': data.median_income,
        'ocean_proximity': data.ocean_proximity
    }])

    prediction = model.predict(user_input)[0]

    return JSONResponse(status_code=200, content={'median_house_value': prediction})