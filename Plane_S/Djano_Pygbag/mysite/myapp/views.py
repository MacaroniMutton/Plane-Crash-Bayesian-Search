from django.shortcuts import render
import subprocess
import os, shutil

# Create your views here.

def exec_pygbag():
    subprocess.call('pygbag --build ProbSims', shell=True, cwd='C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite')
    src = 'C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite\\ProbSims\\build\\web'
    dst = 'C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite\\myapp\\static\\myapp'
    filename = 'probsims.apk'
    shutil.move(os.path.join(src, filename), os.path.join(dst, filename))

def index(request):
    exec_pygbag()
    return render(request, 'myapp/index.html')

