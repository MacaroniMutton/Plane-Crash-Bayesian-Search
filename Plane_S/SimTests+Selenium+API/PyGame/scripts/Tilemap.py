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
        self.grid1 = np.array([[1]*self.columns for _ in range(self.rows)])
        self.grid2 = self.elevation

        for i in range(self.rows):
                for j in range(self.columns):
                    cell = pygame.Rect(j*self.cell_size+1, i*self.cell_size+1, self.cell_size-2, self.cell_size-2)
                    r, g, b, a = self.cmap(self.grid1[i,j]*self.grid2[i,j])
                    r = round(r*255)
                    g = round(g*255)
                    b = round(b*255)
                    self.grid[f"{j};{i}"] = {"tot_prob": self.grid1[i,j]*self.grid2[i,j], "p": self.grid1[i,j], "q":self.grid2[i,j], "rect": cell, "color": (r, g, b)}

    def update(self, searcher):
        x, y = searcher.coords
        searcher_coords = f"{x};{y}"
        self.grid1[y,x] = ((1 - self.grid2[y,x])*self.grid1[y,x]) / (1 - self.grid1[y,x]*self.grid2[y,x])
        self.grid[searcher_coords]["p"] = self.grid1[y,x]
        self.grid[searcher_coords]["tot_prob"] = self.grid1[y,x]*self.grid2[y,x]
        print(self.grid[searcher_coords]["tot_prob"])
        r, g, b, a = self.cmap(self.grid[searcher_coords]["tot_prob"])
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
                r, g, b, a = self.cmap(self.grid[coords]["tot_prob"])
                r = round(r*255)
                g = round(g*255)
                b = round(b*255)
                self.grid[coords]["color"] = (r, g, b)
                  

    def render(self, screen):
        for cell in self.grid.values():
             pygame.draw.rect(screen, cell["color"], cell["rect"])