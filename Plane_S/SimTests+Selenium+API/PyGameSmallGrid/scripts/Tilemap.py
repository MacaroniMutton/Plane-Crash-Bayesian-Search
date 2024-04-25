import pygame
import numpy as np

class Tilemap:
    def __init__(self, game, rows, columns, cell_size, cmap, ocean_depth_np):
        self.game = game
        self.rows = rows
        self.columns = columns
        self.cell_size = cell_size
        self.cmap = cmap
        self.elevation = ocean_depth_np
        self.grid1 = None
        self.grid2 = None
        self.grid = {}
        self.create_base_grid()

    def create_base_grid(self):
        # self.grid1 = np.array([[0]*self.columns for _ in range(self.rows)])
        self.grid1 = np.array([
            [0.02, 0.02, 0.02, 0.01],
            [0.08, 0.17, 0.17, 0.01],
            [0.08, 0.17, 0.17, 0.01],
            [0.02, 0.02, 0.02, 0.01],
        ])
        self.grid2 = self.elevation

        for i in range(self.rows):
                for j in range(self.columns):
                    cell = pygame.Rect(j*self.cell_size+1, i*self.cell_size+1, self.cell_size-2, self.cell_size-2)
                    # print((self.grid1[i,j]*self.grid2[i,j])*6.5)
                    r, g, b, a = self.cmap((self.grid1[i,j]*self.grid2[i,j])*5)
                    r = round(r*255)
                    g = round(g*255)
                    b = round(b*255)
                    self.grid[f"{j};{i}"] = {"tot_prob": self.grid1[i,j]*self.grid2[i,j], "p": self.grid1[i,j], "q":self.grid2[i,j], "rect": cell, "color": (r, g, b)}

    def update(self, searcher):
        print(searcher.coords)
        x, y = searcher.coords
        searcher_coords = f"{x};{y}"
        self.grid1[y,x] = ((1 - self.grid2[y,x])*self.grid1[y,x]) / (1 - self.grid1[y,x]*self.grid2[y,x])
        self.grid[searcher_coords]["p"] = self.grid1[y,x]
        self.grid[searcher_coords]["tot_prob"] = self.grid1[y,x]*self.grid2[y,x]
        totprob = (self.grid[searcher_coords]["tot_prob"])*5
        print(f"{totprob}")
        r, g, b, a = self.cmap((self.grid[searcher_coords]["tot_prob"])*5)
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        self.grid[searcher_coords]["color"] = (r, g, b)
        for coords in self.grid:
            if coords!=searcher_coords:
                j, i = map(int, coords.split(";"))
                self.grid1[i,j] = self.grid1[i,j] / (1 - self.grid1[y,x]*self.grid2[y,x])
                self.grid[coords]["p"] = self.grid1[i,j]
                self.grid[coords]["tot_prob"] = self.grid1[i,j]*self.grid2[i,j]
                r, g, b, a = self.cmap((self.grid[coords]["tot_prob"])*5)
                r = round(r*255)
                g = round(g*255)
                b = round(b*255)
                self.grid[coords]["color"] = (r, g, b)

    # def update(self, searcher):
    #     x, y = searcher.coords
    #     searcher_coords = f"{x};{y}"
    #     self.grid1[y,x] = ((1 - self.grid2[y,x])*self.grid1[y,x]) / (1 - self.grid1[y,x]*self.grid2[y,x])
    #     decrease_factor = self.grid[searcher_coords]["p"] / self.grid1[y,x]
    #     self.grid[searcher_coords]["p"] = self.grid1[y,x]
    #     self.grid[searcher_coords]["tot_prob"] /= decrease_factor
    #     print(f"Searched in the range of latitudes {self.grid[searcher_coords]['lat_range'][0]}-{self.grid[searcher_coords]['lat_range'][1]} and longitudes {self.grid[searcher_coords]['lng_range'][0]}-{self.grid[searcher_coords]['lng_range'][1]} having probability = {self.grid[searcher_coords]['tot_prob']}")
    #     r, g, b, a = self.cmap[round(self.grid[searcher_coords]["tot_prob"], 2)]
    #     r = round(r*255)
    #     g = round(g*255)
    #     b = round(b*255)
    #     self.grid[searcher_coords]["color"] = (r, g, b)
    #     for coords in self.grid:
    #         if coords!=searcher_coords:
    #             j, i = map(int, coords.split(";"))
    #             self.grid1[i,j] = self.grid1[i,j] / (1 - self.grid1[y,x]*self.grid2[y,x])
    #             increase_factor = self.grid1[i,j] / self.grid[coords]["p"]
    #             self.grid[coords]["p"] = self.grid1[i,j]
    #             self.grid[coords]["tot_prob"] *= increase_factor
    #             r, g, b, a = self.cmap[round(min(self.grid[coords]["tot_prob"], 1.0), 2)]
    #             r = round(r*255)
    #             g = round(g*255)
    #             b = round(b*255)
    #             self.grid[coords]["color"] = (r, g, b)
                  

    def render(self, screen):
        for cell in self.grid.values():
             pygame.draw.rect(screen, cell["color"], cell["rect"])