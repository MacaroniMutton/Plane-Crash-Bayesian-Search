import pygame, sys, asyncio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from scipy.stats import norm
from scripts.Tilemap import Tilemap
from scripts.Searcher import Searcher

CELL_SIZE = 8
ROWS = 96
COLUMNS = 96
SEARCHER_COLOR = (255, 0, 0)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CELL_SIZE*COLUMNS, CELL_SIZE*ROWS))
        self.clock = pygame.time.Clock()
        self.cmap = matplotlib.colormaps.get_cmap('coolwarm')
        data = xr.load_dataset('gebco_2023_n15.9082_s11.5225_w69.7148_e74.6367.nc')
        elevation = data.elevation

        li = np.array([[0]*COLUMNS for _ in range(ROWS)])
        mini = float('inf')
        maxi = float('-inf')
        for lat in range(96):
            for lon in range(96):
                li[95-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/96):(lat+1)*int(data.sizes['lat']/96), (lon)*int(data.sizes['lon']/96):(lon+1)*int(data.sizes['lon']/96)].mean().load())
                mini = min(mini, li[95-lat][lon])
                maxi = max(maxi, li[95-lat][lon])

        print(li[-1])
        li = (li-mini)
        li = li/(maxi-mini)
        print(li[-1])

        self.tilemap = Tilemap(self, ROWS, COLUMNS, CELL_SIZE, self.cmap, li)
        self.searcher = Searcher(self, SEARCHER_COLOR, CELL_SIZE)

    async def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.searcher.x_movement[1] = True
                    if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.searcher.x_movement[0] = True
                    if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.searcher.y_movement[1] = True
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.searcher.y_movement[0] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.searcher.x_movement[1] = False
                    if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.searcher.x_movement[0] = False
                    if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.searcher.y_movement[1] = False
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.searcher.y_movement[0] = False

            # self.screen.fill((255,255,255))
            # self.tilemap.render(self.screen)
            # self.searcher.render(self.screen)
            # self.tilemap.update(self.searcher)
            # self.searcher.update(ROWS, COLUMNS)

            self.screen.fill((255,255,255))
            self.tilemap.render(self.screen)
            self.searcher.render(self.screen)
            searcher_coords = self.searcher.coords.copy()
            self.searcher.update(ROWS, COLUMNS)
            new_searcher_coords = self.searcher.coords.copy()
            if searcher_coords!=new_searcher_coords:
                self.tilemap.update(self.searcher)

            pygame.display.update()
            self.clock.tick(10)
            await asyncio.sleep(0)


asyncio.run(Game().run())
