import pygame, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm
from .scripts.bathymetry import li

CELL_SIZE = 10
ROWS = 75
COLUMNS = 84

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CELL_SIZE*COLUMNS, CELL_SIZE*ROWS))
        self.clock = pygame.time.Clock()
        self.x_movement = [False, False]
        self.y_movement = [False, False]
        self.coords = [20*CELL_SIZE+1, 20*CELL_SIZE+1]
        self.player_rect = pygame.Rect(self.coords[0], self.coords[1], CELL_SIZE-2, CELL_SIZE-2)
    

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.x_movement[1] = True
                    if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.x_movement[0] = True
                    if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.y_movement[1] = True
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.y_movement[0] = True
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.x_movement[1] = False
                    if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.x_movement[0] = False
                    if event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        self.y_movement[1] = False
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        self.y_movement[0] = False

            self.screen.fill((255,255,255))
            for i in range(ROWS):
                for j in range(COLUMNS):
                    cell = pygame.Rect(j*CELL_SIZE+1, i*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2)
                    pygame.draw.rect(self.screen, (0,0,255), cell)
            
            
            cmap = matplotlib.colormaps.get_cmap('coolwarm')
            # for i in range(len(Z)):
            #     for j in range(len(Z[i])):
            #         cell = pygame.Rect(i*CELL_SIZE+1, j*CELL_SIZE+1, CELL_SIZE-2, CELL_SIZE-2)
            #         r,g,b,a = cmap(Z[i][j])
            #         pygame.draw.rect(self.screen, (r*255,g*255,b*255), cell)



            self.coords[0] += (self.x_movement[1] - self.x_movement[0])*CELL_SIZE
            self.coords[1] += (self.y_movement[1] - self.y_movement[0])*CELL_SIZE
            self.player_rect.x = self.coords[0]
            self.player_rect.y = self.coords[1]
            pygame.draw.rect(self.screen, (255,0,0), self.player_rect)
            pygame.display.update()
            self.clock.tick(7)

Game().run()
