from django.contrib import admin
from django.urls import path, include
from . import views

app_name = "myapp"
urlpatterns = [
    path("", views.index, name="index"),
    path("services/", views.services, name="services"),
    path("proceed_info/", views.flightInfo, name="flightInfo"),
]