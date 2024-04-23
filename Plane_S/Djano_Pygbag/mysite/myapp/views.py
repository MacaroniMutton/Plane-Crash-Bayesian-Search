from django.shortcuts import render
import subprocess
import os, shutil
from django.shortcuts import render
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pickle

# Create your views here.


file_path = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite\\ProbSims\\report.txt"

def exec_pygbag():
    subprocess.call('pygbag --build ProbSims', shell=True, cwd='C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite')
    src = 'C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite\\ProbSims\\build\\web'
    dst = 'C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag\\mysite\\myapp\\static\\myapp'
    filename = 'probsims.apk'
    shutil.move(os.path.join(src, filename), os.path.join(dst, filename))


# Define a custom event handler to handle file system events
class MyEventHandler(FileSystemEventHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def on_modified(self, event):
        if event.src_path == os.path.abspath(file_path):
            # File modified event detected, trigger the callback
            coordinates = self.read_coordinates()
            self.callback(coordinates)

    def read_coordinates(self):
        try:
            with open(file_path, 'rb') as f:
                coordinates = pickle.load(f)
                f.close()
                print(coordinates)
                return coordinates
        except Exception as e:
            print(f"Error reading coordinates file: {e}")
            return None

# Function to render the template with updated data
def render_template_with_coordinates(request, coordinates):
    context = {'coordinates': coordinates}
    return render(request, 'myapp/index.html', context)

# Function to start file monitoring
def start_file_monitoring(callback):
    event_handler = MyEventHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(file_path)), recursive=False)
    observer.start()

# Function to stop file monitoring
def stop_file_monitoring(observer):
    observer.stop()
    observer.join()


def index(request):
    # Define the callback function to render the template with updated coordinates
    def callback(coordinates):
        return render_template_with_coordinates(request, coordinates)

    # Start file monitoring
    start_file_monitoring(callback)

    # Execute your pygbag function here (ensure the file is updated)
    exec_pygbag()

    # Return an initial render of the template (this will be updated by the callback)
    return render(request, 'myapp/index.html')



