from math import sin, cos, pi
import numpy as np
import pygame


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
        self.gridprod = np.array([[0. for _ in range(self.rows)] for _ in range(self.columns)])
        self.grid = {}
        self.create_base_grid()

    def makeGaussian(self, size, fwhm = 4, center=None):
        """ Make a square gaussian kernel.

        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """

        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        theta = (110)*pi/180
        a = 40
        b = 20
        x0 = 10
        y0 = 70

        # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        return np.exp(-4*np.log(2) * (((x-x0)*cos(theta)-(y-y0)*sin(theta))**2 / a**2 + ((y-y0)*cos(theta)+(x-x0)*sin(theta))**2 / b**2) )

    def create_base_grid(self):
        self.grid1 = self.makeGaussian(96)
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
                    self.grid[f"{j};{i}"] = {"tot_prob": self.gridprod[i, j], "p": self.grid1[i,j], "q":self.grid2[i,j], "rect": cell, "color": (r, g, b)}

    def update(self, searcher):
        x, y = searcher.coords
        searcher_coords = f"{x};{y}"
        self.grid1[y,x] = ((1 - self.grid2[y,x])*self.grid1[y,x]) / (1 - self.grid1[y,x]*self.grid2[y,x])
        self.grid[searcher_coords]["p"] = self.grid1[y,x]
        self.grid[searcher_coords]["tot_prob"] = self.grid1[y,x]*self.grid2[y,x]
        print(self.grid[searcher_coords]["tot_prob"])
        r, g, b, a = self.cmap[round(800*self.grid[searcher_coords]["tot_prob"], 2)]
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
                r, g, b, a = self.cmap[round(800*self.grid[coords]["tot_prob"], 2)]
                r = round(r*255)
                g = round(g*255)
                b = round(b*255)
                self.grid[coords]["color"] = (r, g, b)
                  

    def render(self, screen):
        for cell in self.grid.values():
             pygame.draw.rect(screen, cell["color"], cell["rect"])