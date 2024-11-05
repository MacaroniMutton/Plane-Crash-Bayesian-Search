# Optimizing Search for Lost Objects in Water Bodies

**Project Description:**  
"Optimizing Search for Lost Objects in Water Bodies" is a web-based tool designed to assist in locating objects such as airplanes and ships lost in vast water bodies. This project utilizes **Bayesian Search Theory** and the **A\* Search Algorithm** to create probability-based search paths, enhancing the efficiency and success rate of search and rescue operations.

## Introduction

This project is built to streamline and improve search operations for lost objects in oceans by calculating probability distributions around last-known locations and observed debris. The web app incorporates **Bayesian updates** and **optimal path simulations** to suggest effective search paths, integrating real-time environmental data such as ocean currents and wind direction.

## Features

- **Automated Data Collection**: Uses Selenium to gather bathymetric data.
- **Dynamic Probability Updates**: Bayesian theory allows probabilities to adjust as new data is received.
- **Optimal Search Path Simulation**: Uses A* and probability distributions to calculate effective search paths.
- **Downloadable Search Paths**: Enables downloading optimal search routes in CSV format.
- **User-Friendly Interface**: Simplified UI for inputting data and visualizing search outcomes.

## System Workflow

1. **User Authentication**: Search and rescue personnel log in to access the service.
2. **Input Data**: Flight data, last-known location, and debris recovery details are entered.
3. **Data Processing**: Bayesian and A* algorithms create probability distributions and optimal search paths.
4. **Simulation**: Users view the simulated search path in a graphical interface.
5. **Data Download**: Optimal paths and probabilities can be downloaded in CSV format for operational use.

## Tech Stack

- **Backend**: Django, SQLite
- **Frontend**: HTML, CSS, JavaScript, Leaflet JS, Turf JS
- **Python Libraries**: NumPy, Pandas, Matplotlib, Scikit-Learn
- **GUI**: Pygame, Pygbag
- **Automation**: Selenium
- **APIs Used**: 
   - Meteomatics (ocean current and wind data)
   - Geocoding (city location data)

## Algorithms Used

- **Bayesian Search Theory**: Enables continuous probability updates for search areas.
- **A* Search Algorithm**: Calculates the most efficient search paths based on probability grids.
- **Kernel Density Estimation**: Applies unsupervised learning to generate reverse drift distributions based on debris data.

## Usage

1. **Log in** to the application.
2. **Input flight and debris recovery data**.
3. **Run the simulation** to visualize the optimal search path.
4. **Download the search path CSV** for use in field operations.

## Future Scope

- **Expanded Search Area**: Broaden the current search grid and allow users to input historical data.
- **Collaborative Features**: Enable data and strategy sharing among users.
- **Advanced Analytics**: Add predictive modeling and anomaly detection to enhance search accuracy.
- **Commercialization**: Deploy the platform as a service for government and private agencies.

## Contributors

- Anushka Jadhav
- Atharva Jagtap
- Manan Kher

## Acknowledgments

Special thanks to our guides, **Prof. Anand Godbole** and **Mr. Raj Mehta**, for their guidance and support.
