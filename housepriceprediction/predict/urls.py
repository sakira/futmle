# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 09:56:33 2018

@author: hassans
"""

from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train', views.train_model, name='train')
]