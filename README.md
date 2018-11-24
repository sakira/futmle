# futmle
housing prices prediction on city districts using regression


The task is to setup a web service for predicting housing prices c
on 
city districts using regression and to deploy the service publicly 

(for example, on heroku.com). 
The service should offer an endpoint that 
takes as input certain 
statistics of the district (defined below), and 

outputs the estimated price of a dwelling.

-------------------------------------------------
# Exploratory data analysis
------------------------------------------------
See housingpriceprediction/kernel.ipynb



---------------------------------------------------------------------
# Setup environment
---------------------------------------------------------------------
1. Create your own environment in conda
conda create -n yourenv<py37myenv> python=3.7
source activate yourenv<py37myenv> 

2. Installation of Django (version 2.1.2)
conda install -n py37myenv django
python -m django --version

3. Create the project in a particular directory (change directory path if necessary)

django-admin startproject housepriceprediction
cd housepriceprediction
python manage.py runserver

4. Test to see if the webservice is working correctly

Open a browser
Type localhost:8000

5. Creating a predict app
python manage.py startapp predict

6. Write the codes in predict/views.py
7. Create the file predict/urls.py for mapping to a URL
8. Point the root URLconf at the predict.urls module in file housepriceprediction/urls.py
9. Run the server
python manage.py runserver
In browser, open localhost:8000/predict/


-----------------------------------------------------------------------
# How to use the App locally
-----------------------------------------------------------------------
predict/views.py contains functions for prediction app and training app.
Open the browser and type for example:
1. To train the model on the data stored in the server: 
localhost:8000/train 
2. To predict from the trained model:
curl http://localhost:8000/predict/ -H application/json --data-binary '{
  "crime_rate": 0.1,
  "avg_number_of_rooms": 4.0,
  "distance_to_employment_centers": 6.5,
  "property_tax_rate": 330.0,
  "pupil_teacher_ratio": 19.5
}'
-----------------------------------------------------------------------

------------------------------------------------------------------------
# How to use the App remotely from another server
------------------------------------------------------------------------

1. Example request for training with the given data:
curl http://sakira.pythonanywhere.com/predict/train
2. Example request for prediction from a json object:
   curl http://sakira.pythonanywhere.com/predict/ -H application/json --data-binary '{
  "crime_rate": 0.9557700000000001,
  "avg_number_of_rooms": 6.047,
  "distance_to_employment_centers": 4.4534,
  "property_tax_rate": 307.0,
  "pupil_teacher_ratio": 21.0
}'
