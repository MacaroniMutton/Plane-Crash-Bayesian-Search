import pickle
import os
import numpy as np
import pandas as pd
# import matplotlib.colormaps
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import xarray as xr
from .scripts.Tilemap import Tilemap
from .scripts.Searcher import Searcher

CELL_SIZE = 8
ROWS = 96
COLUMNS = 96
SEARCHER_COLOR = (255, 0, 0)

class Game:
    def __init__(self):
        self.cmap = None
        with open('C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\ProbSims\\distributions_data.pkl', 'rb') as fp:
                distributions_data = pickle.load(fp)
        

        bathy_li = distributions_data['bathy_li']
        gauss_distr = distributions_data['gaussian']
        lat_lng_li = distributions_data['lat_lng_li']
        circular_dist = distributions_data['circular_uniform']
        self.lkp_latitude = distributions_data['lkp_latitude']
        self.lkp_longitude = distributions_data['lkp_longitude']
        gauss_distr = gauss_distr/gauss_distr.sum()
        circular_dist = circular_dist/circular_dist.sum()
        first_dist = gauss_distr*0.5 + circular_dist*0.5
        circular_dist[circular_dist==0] = 1e-35
        if "shrinked_rd_dist" in distributions_data.keys():
            shrinked_rd_dist = distributions_data['shrinked_rd_dist']
            shrinked_rd_dist = shrinked_rd_dist/shrinked_rd_dist.sum()
            shrinked_rd_dist[shrinked_rd_dist==0] = 1e-35
            # shrinked_rd_dist = shrinked_rd_dist * 2
            final_dist = first_dist*0.7 + 0.3*shrinked_rd_dist
        else:
            final_dist = first_dist
        

        self.plane_coords = [30, 40]
        self.tilemap = Tilemap(self, ROWS, COLUMNS, CELL_SIZE, self.cmap, bathy_li, final_dist, lat_lng_li, self.plane_coords)
        self.searcher = Searcher(self, SEARCHER_COLOR, CELL_SIZE, [0,0])
        self.report_df = pd.DataFrame({'lat_bottom':[], 'lat_top':[], 'lon_left':[], 'lon_right':[], 'probability':[]})

    def grid_has_value_greater_than(self, k):

        found_greater = False
        
        for cell, cell_data in self.tilemap.grid.items():
            value = cell_data['tot_prob']
            # Check if the current value is greater than k
            if value > k:
                # Update the flag and break out of the inner loop
                found_greater = True
                break
            
            # Check the flag after each inner loop iteration
            if found_greater:
                break  # Break out of the outer loop if a value > k is found
        
        return found_greater

    def run(self):
        # with open("ProbSimsReportGen\\report.txt", "w") as fp:
        #     fp.write("")
        while True:

            path_to_max_prob = self.searcher.search_highest_probability_square()
            searcher_coords = self.searcher.coords.copy()
            str_searcher_coords = f"{searcher_coords[0]};{searcher_coords[1]}"
            self.searcher.update(path_to_max_prob)
            new_searcher_coords = self.searcher.coords.copy()
            if searcher_coords!=new_searcher_coords or len(path_to_max_prob)==1:
                self.tilemap.update(self.searcher)
                print(f"Search in the range of latitudes {self.tilemap.grid[str_searcher_coords]['lat_range'][0]}-{self.tilemap.grid[str_searcher_coords]['lat_range'][1]} and longitudes {self.tilemap.grid[str_searcher_coords]['lng_range'][0]}-{self.tilemap.grid[str_searcher_coords]['lng_range'][1]} having probability = {self.tilemap.grid[str_searcher_coords]['tot_prob']}")
                self.report_df.loc[len(self.report_df.index)] = [self.tilemap.grid[str_searcher_coords]['lat_range'][0], self.tilemap.grid[str_searcher_coords]['lat_range'][1], self.tilemap.grid[str_searcher_coords]['lng_range'][0], self.tilemap.grid[str_searcher_coords]['lng_range'][1], self.tilemap.grid[str_searcher_coords]['tot_prob']]
                k = 0.2
                if self.grid_has_value_greater_than(k):
                    pass
                else:
                    break

                
                # try:
                #     # with open("ProbSimsReportGen\\report.txt", "a") as fp:
                #         # fp.write(f"Search in the range of latitudes {self.tilemap.grid[str_searcher_coords]['lat_range'][0]}-{self.tilemap.grid[str_searcher_coords]['lat_range'][1]} and longitudes {self.tilemap.grid[str_searcher_coords]['lng_range'][0]}-{self.tilemap.grid[str_searcher_coords]['lng_range'][1]} having probability = {self.tilemap.grid[str_searcher_coords]['tot_prob']}\n")
                #     #     print(f"Searcher coords : {self.searcher.coords}")
                #     #     print("Successfully dumped coordinates to report.txt")

                #     # with open("ProbSimsReportGen\\report.txtt", "r") as fp:
                #     #     print(fp.readlines())

                # except Exception as e:
                #     print(f"Error while dumping coordinates to report.txt: {e}")
        
        return self.report_df



