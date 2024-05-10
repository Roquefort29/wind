from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import pandas as pd
from keras.models import load_model

def index(request):
    return render(request, 'turbo/index.html')
