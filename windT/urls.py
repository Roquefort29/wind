from django.urls import path

from .views import *

urlpatterns = [
    path('', index, name='home'),
    path('contact', contacts, name='contacts'),
    path('predict', predict, name='predict'),
    path('countries/', country_list, name='country_list'),
    path('country/<int:country_id>/', country_detail, name='country_detail')
]
