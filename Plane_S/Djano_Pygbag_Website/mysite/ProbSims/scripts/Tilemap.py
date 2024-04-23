from math import sin, cos, pi
import numpy as np
import pygame


class Tilemap:
    def __init__(self, game, rows, columns, cell_size, cmap, ocean_depth_np, gauss_distr, lat_lng_li, plane_coords):
        self.game = game
        self.rows = rows
        self.columns = columns
        self.cell_size = cell_size
        self.cmap = cmap
        self.elevation = ocean_depth_np
        self.gauss_distr = gauss_distr
        self.lat_lng_li = lat_lng_li
        self.plane_coords = plane_coords
        self.grid1 = None
        self.grid2 = None
        self.gridprod = np.array([[0. for _ in range(self.rows)] for _ in range(self.columns)])
        self.grid = {}
        self.create_base_grid()


    def create_base_grid(self):
        self.grid1 = self.gauss_distr
        self.grid1 = self.grid1 / self.grid1.sum()
        self.grid2 = self.elevation
        for i in range(self.rows):
            for j in range(self.columns):
                self.gridprod[i, j] = 100000*self.grid1[i,j]*self.grid2[i,j]
        mini = self.gridprod.min()
        maxi = self.gridprod.max()
        self.gridprod = (self.gridprod-mini)
        self.gridprod = self.gridprod/(maxi-mini)
        for i in range(self.rows):
                for j in range(self.columns):
                    cell = pygame.Rect(j*self.cell_size+1, i*self.cell_size+1, self.cell_size-2, self.cell_size-2)
                    r, g, b, a = self.cmap[round(self.gridprod[i,j], 2)]
                    r = round(r*255)
                    g = round(g*255)
                    b = round(b*255)
                    lat_lng = self.lat_lng_li[i,j]
                    lat_range, lng_range = lat_lng
                    if j==self.plane_coords[0] and i==self.plane_coords[1]:
                        self.grid[f"{j};{i}"] = {"tot_prob": self.gridprod[i, j], "p": self.grid1[i,j], "q":self.grid2[i,j], "rect": cell, "color": (r, g, b), "lat_range": lat_range, "lng_range": lng_range, "plane_present": True}
                    else:
                        self.grid[f"{j};{i}"] = {"tot_prob": self.gridprod[i, j], "p": self.grid1[i,j], "q":self.grid2[i,j], "rect": cell, "color": (r, g, b), "lat_range": lat_range, "lng_range": lng_range, "plane_present": False}

    def update(self, searcher):
        x, y = searcher.coords
        searcher_coords = f"{x};{y}"
        self.grid1[y,x] = ((1 - self.grid2[y,x])*self.grid1[y,x]) / (1 - self.grid1[y,x]*self.grid2[y,x])
        decrease_factor = self.grid[searcher_coords]["p"] / self.grid1[y,x]
        self.grid[searcher_coords]["p"] = self.grid1[y,x]
        self.grid[searcher_coords]["tot_prob"] /= decrease_factor
        print(f"Searched in the range of latitudes {self.grid[searcher_coords]['lat_range'][0]}-{self.grid[searcher_coords]['lat_range'][1]} and longitudes {self.grid[searcher_coords]['lng_range'][0]}-{self.grid[searcher_coords]['lng_range'][1]} having probability = {self.grid[searcher_coords]['tot_prob']}")
        r, g, b, a = self.cmap[round(self.grid[searcher_coords]["tot_prob"], 2)]
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        self.grid[searcher_coords]["color"] = (r, g, b)
        for coords in self.grid:
            if coords!=searcher_coords:
                j, i = map(int, coords.split(";"))
                self.grid1[i,j] = self.grid1[i,j] / (1 - self.grid1[y,x]*self.grid2[y,x])
                increase_factor = self.grid1[i,j] / self.grid[coords]["p"]
                self.grid[coords]["p"] = self.grid1[i,j]
                self.grid[coords]["tot_prob"] *= increase_factor
                r, g, b, a = self.cmap[round(min(self.grid[coords]["tot_prob"], 1.0), 2)]
                r = round(r*255)
                g = round(g*255)
                b = round(b*255)
                self.grid[coords]["color"] = (r, g, b)
                  

    def render(self, screen):
        for cell in self.grid.values():
             pygame.draw.rect(screen, (cell["color"][0], cell["color"][1], cell["color"][2], 120), cell["rect"])