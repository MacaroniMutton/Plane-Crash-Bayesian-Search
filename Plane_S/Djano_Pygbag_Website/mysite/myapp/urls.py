from django.contrib import admin
from django.urls import path, include
from . import views

app_name = "myapp"
urlpatterns = [
    path("", views.index, name="index"),
    path("services/", views.services, name="services"),
    path("proceed_info/", views.flightInfo, name="flightInfo"),
    path("rd/", views.rd_view, name="rd"),
    path("game/", views.game, name="game"),
    path("contact/", views.contact, name="contact"),
    path('download-csv/', views.download_dataframe_as_csv, name='download_csv'),
    path('about/', views.about, name="about"),
    path('logout/', views.logout_view, name="logout"),
]
