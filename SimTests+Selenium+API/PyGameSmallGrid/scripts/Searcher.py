import pygame

class Searcher:
    def __init__(self, game, color, size):
        self.game = game
        self.color = color
        self.size = size
        self.x_movement = [False, False]
        self.y_movement = [False, False]
        self.coords = [0,0]
        self.player_rect = pygame.Rect(self.coords[0]*self.size+1, self.coords[1]*self.size+1, self.size-2, self.size-2)

    def update(self, rows, cols):
        if (self.x_movement[1] - self.x_movement[0])>0 and self.coords[0]<cols-1:
            self.coords[0] += (self.x_movement[1] - self.x_movement[0])
        if (self.x_movement[1] - self.x_movement[0])<0 and self.coords[0]>0:
            self.coords[0] += (self.x_movement[1] - self.x_movement[0])
        if (self.y_movement[1] - self.y_movement[0])>0 and self.coords[1]<rows-1:
            self.coords[1] += (self.y_movement[1] - self.y_movement[0])
        if (self.y_movement[1] - self.y_movement[0])<0 and self.coords[1]>0:
            self.coords[1] += (self.y_movement[1] - self.y_movement[0])
        self.player_rect.x = self.coords[0]*self.size+1
        self.player_rect.y = self.coords[1]*self.size+1

    def render(self, screen):
        pygame.draw.rect(screen, self.color, self.player_rect)