from django.urls import path

from .views import *

urlpatterns = [
    path('', index, name='home'),
    path('contact', contacts, name='contacts'),
    path('predict', predict, name='predict')
]
