from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.core.exceptions import SuspiciousOperation

from django.views.decorators.csrf import csrf_exempt




from predict import train

import json

# This function takes data as json format and 
# predict from the trained model
# The output is the estimated house price and
# The standard deviation of it.
@csrf_exempt # ignore token for now
def index(request):
    
    
    # if the request is a POST
    if request.POST:
        response = train.valid_data(request)
        if response == False:
            #raise  SuspiciousOperation('400 Bad request....')
            return HttpResponse('400 Bad request....')
        
        response_data = {}
        response_data["housing_value"] = response[0]
        print(response)
        print(response_data)
        return HttpResponse(json.dumps(response_data), content_type="application/json")
        #train.predict(1)
        
        
    #print(request.POST)
    
    else:
        return HttpResponse("Helloworld!")
    
    
    return HttpResponse("OK")


# Train the model. Right place for the function is in 'models.py'   
def train_model(request):
    
    try:
        train.train_model()
        return HttpResponse("Trained")
    except:
        return HttpResponse("Error occured")

