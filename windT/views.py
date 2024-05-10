from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return render(request, 'turbo/index.html')

def contacts(request):
    return render(request, 'turbo/contacts.html')

def predict(request):
    return render(request, 'turbo/predict.html')
