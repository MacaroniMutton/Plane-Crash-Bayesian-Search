from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.http import JsonResponse
from .city_api import city_lat_lng

# Create your views here.


def index(request):
    if request.method == "POST":
        form_id = request.POST.get('form_id')

        if form_id == "register":
            username = request.POST.get("username")
            email = request.POST.get("email")
            password1 = request.POST.get("password1")
            password2 = request.POST.get("password2")

            if User.objects.filter(username=username).exists():
                return JsonResponse({'success': False, 'error_message': 'Username already exists.'})
            elif User.objects.filter(email=email).exists():
                return JsonResponse({'success': False, 'error_message': 'Email address already exists.'})
            elif password1 != password2:
                return JsonResponse({'success': False, 'error_message': 'Passwords do not match.'})
            else:
                user = User.objects.create_user(username=username, email=email, password=password1)
                login(request, user)
                return JsonResponse({'success': True})

        elif form_id == "login":
            username = request.POST.get("username").lower()
            password = request.POST.get("password")
            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)
                return JsonResponse({'success': True})
            else:
                return JsonResponse({'success': False, 'error_message': 'Username or password is incorrect.'})

    return render(request, 'myapp/index.html')


def services(request):
    lat = None
    lng = None

    if request.method == 'POST':
        form_id = request.POST.get('form_id')
        print(form_id)
        if form_id == "sourceCity":
            print(request.POST)
            city = request.POST.get("sourceCity")
            print(city)
            lat, lng = city_lat_lng(city)
            return JsonResponse({'latitude': lat, 'longitude': lng})

        elif form_id == "destCity":
            city = request.POST.get("destCity")
            lat, lng = city_lat_lng(city)
            lat, lng = city_lat_lng(city)
            return JsonResponse({'latitude': lat, 'longitude': lng})
    
    # Render the services.html template with latitude and longitude
    return render(request, "myapp/services.html", {'latitude': lat, 'longitude': lng})
