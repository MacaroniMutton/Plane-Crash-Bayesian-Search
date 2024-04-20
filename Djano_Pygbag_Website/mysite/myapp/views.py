from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.http import JsonResponse
from .utils import *
import json
from .models import FlightInfo
import os

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
        
        elif form_id== "proceedButton":
            source_latitude = request.POST.get('source_latitude')
            source_longitude = request.POST.get('source_longitude')
            dest_latitude = request.POST.get('dest_latitude')
            dest_longitude = request.POST.get('dest_longitude')
            next_latitude = request.POST.get('next_latitude')
            next_longitude = request.POST.get('next_longitude')
            flight_path = request.POST.get('flight_path')
            flight_path = json.loads(flight_path)
            lkp_latitude = request.POST.get('lkp_latitude')
            lkp_latitude = float(lkp_latitude)
            lkp_longitude = request.POST.get('lkp_longitude')
            lkp_longitude = float(lkp_longitude)
            context = {
                'source_latitude': source_latitude,
                'source_longitude': source_longitude,
                'dest_latitude': dest_latitude,
                'dest_longitude': dest_longitude,
                'next_latitude': next_latitude,
                'next_longitude': next_longitude,
                'flight_path': flight_path,
                'lkp_latitude': lkp_latitude,
                'lkp_longitude': lkp_longitude,
            }

            myJsonDict = json.dumps(context)
            FlightInfo.objects.filter(user=request.user).delete()
            FlightInfo.objects.create(user=request.user, context=myJsonDict)

            distances_from_lkp = [haversine_distance(lkp_latitude, lkp_longitude, point["lat"], point["lng"]) for point in flight_path]
            print(distances_from_lkp)

            min_distance_index = distances_from_lkp.index(min(distances_from_lkp))

            nearest_point = flight_path[min_distance_index]
            print(nearest_point)

            closer_point = None

            before_index = min_distance_index - 1
            after_index = min_distance_index + 1

            # Check if before_index is valid and calculate distance to before_point
            if before_index >= 0:
                distance_to_before = distances_from_lkp[before_index]
                before_point = flight_path[before_index]
            else:
                distance_to_before = float('inf')  # Set a large value for invalid index
                before_point = None

            # Check if after_index is valid and calculate distance to after_point
            if after_index < len(flight_path):
                distance_to_after = distances_from_lkp[after_index]
                after_point = flight_path[after_index]
            else:
                distance_to_after = float('inf')  # Set a large value for invalid index
                after_point = None

            # Compare distances to determine which point is closer
            if distance_to_before <= distance_to_after:
                closer_point = before_point
            else:
                closer_point = after_point

            if closer_point == before_point:
                angle = calculate_vector_angle(closer_point, nearest_point)
            else:
                angle = calculate_vector_angle(nearest_point, closer_point)
            print(angle)

            # pass angle to makeGaussian, get the normal dist
            gauss_distr = makeGaussian(96, 25, 13, angle)

            # Call a function which downloads the bathymetry data using Selenium
            download_bathymetry_data(lkp_latitude, lkp_longitude)
            # Specify the path to the zip file
            zip_file_path = None
            base_directory = r"C:\Users\Manan Kher\OneDrive\Documents\Plane-Crash-Bayesian-Search\Djano_Pygbag_Website\mysite\myapp"
            files_in_directory = os.listdir(base_directory)
            matching_files = [filename for filename in files_in_directory if filename.startswith("GEBCO")]
            if matching_files:
                zip_filename = matching_files[0]
                zip_file_path = os.path.join(base_directory, zip_filename)

            # Specify the directory where you want to extract the contents
            extract_to_directory = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\Plane-Crash-Bayesian-Search\\Djano_Pygbag_Website\\mysite\\myapp\\BATHY"

            # Create the extraction directory if it does not exist
            os.makedirs(extract_to_directory, exist_ok=True)
        
            # Extract the zip file
            extract_zip(zip_file_path, extract_to_directory)
            zip_folder = os.path.basename(zip_file_path)
            extracted_folder = zip_folder.split('.')[0]

            print(f"Zip file extracted to: {extract_to_directory}")

            nc_file_path = find_gebco_nc_file(os.path.join(extract_to_directory, extracted_folder))

            li = load_bathymetry_data(nc_file_path)

            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)


            # Call function which returns reverse drift dist, uses api calls to current and wind, taking input lat lon time
            

            # Combine all three distributions into final distribution

            # Write the distribution to a file, which will be read by main.py

            # return redirect('myapp:game')

            # deg game(request): exec_pygbag   return render()
            # import exec_pygbag from utils.py
    
    return render(request, "myapp/services.html", {'latitude': lat, 'longitude': lng})


def flightInfo(request):
    flight = FlightInfo.objects.filter(user=request.user)
    context = json.loads(flight[0].context)
    return render(request, 'myapp/proceed_info.html', context)


# def proceed_view(request):
#     if request.method == 'POST' and request.is_ajax():
#         source_latitude = request.POST.get('source_latitude')
#         source_longitude = request.POST.get('source_longitude')
#         dest_latitude = request.POST.get('dest_latitude')
#         dest_longitude = request.POST.get('dest_longitude')
#         flight_path = request.POST.get('flight_path')
#         lkp_latitude = request.POST.get('lkp_latitude')
#         lkp_longitude = request.POST.get('lkp_longitude')

#         # Process the data as needed (e.g., save to database, perform calculations, etc.)

#         # Return a JSON response with a redirect URL
#         response_data = {
#             'redirect_url': '/path/to/redirect'  # Specify the URL to redirect to after processing
#         }
#         return JsonResponse(response_data)

#     # Handle invalid requests or non-ajax requests
#     return JsonResponse({'error': 'Invalid request'}, status=400)
