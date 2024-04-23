import pygame
import heapq

class Searcher:
    def __init__(self, game, color, size, start_coords):
        self.game = game
        self.grid = self.game.tilemap.grid
        self.grid_size = self.game.tilemap.rows
        self.color = color
        self.size = size
        self.x_movement = [False, False]
        self.y_movement = [False, False]
        self.coords = start_coords
        self.player_rect = pygame.Rect(self.coords[0]*self.size+1, self.coords[1]*self.size+1, self.size-2, self.size-2)

    # def update(self, rows, cols):
    #     if (self.x_movement[1] - self.x_movement[0])>0 and self.coords[0]<cols-1:
    #         self.coords[0] += (self.x_movement[1] - self.x_movement[0])
    #     if (self.x_movement[1] - self.x_movement[0])<0 and self.coords[0]>0:
    #         self.coords[0] += (self.x_movement[1] - self.x_movement[0])
    #     if (self.y_movement[1] - self.y_movement[0])>0 and self.coords[1]<rows-1:
    #         self.coords[1] += (self.y_movement[1] - self.y_movement[0])
    #     if (self.y_movement[1] - self.y_movement[0])<0 and self.coords[1]>0:
    #         self.coords[1] += (self.y_movement[1] - self.y_movement[0])
    #     self.player_rect.x = self.coords[0]*self.size+1
    #     self.player_rect.y = self.coords[1]*self.size+
    
    def update(self, path_to_max_prob):
        if len(path_to_max_prob)>0:
            self.coords = list(path_to_max_prob[1])
        else:
            self.coords = list(path_to_max_prob[0])
        self.player_rect.x = self.coords[0]*self.size+1
        self.player_rect.y = self.coords[1]*self.size+1

    def render(self, screen):
        pygame.draw.rect(screen, self.color, self.player_rect)

    def distance_between(self, cell1, cell2):
        cell2_coords = f"{cell2[0]};{cell2[1]}"
        # Calculate cost based on inverse probability of cell2
        probability = min(self.grid[cell2_coords]["tot_prob"],1)
        if probability > 0:
            return 1 / probability
        else:
            # Handle zero or near-zero probabilities (e.g., set a high cost)
            return float('inf')  # Infinite cost for impassable cells

    def heuristic_cost(self, current, goal):
        # Manhattan distance
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def astar_search(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))  # (f-value, node)
        g_scores = {start: 0}
        parent_map = {start: None}

        while open_set:
            current_f, current_node = heapq.heappop(open_set)

            if current_node == goal:
                # Reconstruct path
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = parent_map[current_node]
                return path[::-1]

            for neighbor in self.get_neighbors(current_node):
                tentative_g = g_scores[current_node] + self.distance_between(current_node, neighbor)

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_value = tentative_g + self.heuristic_cost(neighbor, goal)
                    heapq.heappush(open_set, (f_value, neighbor))
                    parent_map[neighbor] = current_node

        return None

    def get_neighbors(self, node):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        neighbors = [(node[0] + dx, node[1] + dy) for dx, dy in directions]
        return [n for n in neighbors if 0 <= n[0] < self.grid_size and 0 <= n[1] < self.grid_size]
    
    def search_highest_probability_square(self):
        max_prob = -1
        max_prob_square = None

        for coords, cell_data in self.grid.items():
            if cell_data["tot_prob"] > max_prob:
                max_prob = cell_data["tot_prob"]
                max_prob_square = coords

        if not max_prob_square:
            return None

        max_prob_square = tuple(map(int, max_prob_square.split(";")))
        start_node = tuple(self.coords)
        goal_node = max_prob_square
        path = self.astar_search(start_node, goal_node)

        return path

    