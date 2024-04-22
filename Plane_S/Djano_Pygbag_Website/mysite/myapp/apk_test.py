from androguard.core.apk import APK
import pickle

def view_apk_contents(apk_path):
    # Load the APK file
    apk = APK(apk_path)

    # Extract basic information
    print(f"Package Name: {apk.get_package()}")


    # Print activities
    print("\nActivities:")
    for activity in apk.get_activities():
        print(activity)

    # Print permissions
    print("\nPermissions:")
    for permission in apk.get_permissions():
        print(permission)

    # Print services
    print("\nServices:")
    for service in apk.get_services():
        print(service)

    # Print receivers
    print("\nReceivers:")
    for receiver in apk.get_receivers():
        print(receiver)

    # Print providers
    print("\nContent Providers:")
    for provider in apk.get_providers():
        print(provider)

    # Print libraries
    print("\nLibraries:")
    for lib in apk.get_libraries():
        print(lib)

    # Print files in the APK
    print("\nFiles in the APK:")
    for filename in apk.get_files():
        print(filename)
    
    print(apk.get_file("assets/main.py").decode('utf-8'))

def extract_and_read_report_pkl(apk_path):
    # Load the APK file
    apk = APK(apk_path)

    # Specify the path to the report.pkl file within the APK
    report_pkl_path = "assets/report.pkl"

    report = apk.get_file(report_pkl_path)

    

    print()
    print(report)


if __name__ == "__main__":
    # Specify the path to your APK file
    apk_file_path = "C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\myapp\\static\\myapp\\probsims.apk"

    # Call the function to view APK contents
    view_apk_contents(apk_file_path)


    # Call the function to extract and read report.pkl
    extract_and_read_report_pkl(apk_file_path)
