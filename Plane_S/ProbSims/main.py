import pickle
import numpy as np
# import matplotlib.colormaps
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import xarray as xr
from scripts.Tilemap import Tilemap
from scripts.Searcher import Searcher
import pygame, asyncio

CELL_SIZE = 7
ROWS = 96
COLUMNS = 96
SEARCHER_COLOR = (255, 0, 0)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CELL_SIZE*COLUMNS+100, CELL_SIZE*ROWS+100))
        self.screen_height = CELL_SIZE*ROWS+100
        self.screen_width = CELL_SIZE*COLUMNS+100
        self.clock = pygame.time.Clock()
        self.cmap = {0.0: (0.2298057, 0.298717966, 0.753683153, 1.0), 0.01: (0.2389484589019608, 0.3123654946588235, 0.7656759021764705, 1.0), 0.02: (0.2526625972549019, 0.3328367876470588, 0.7836650259411765, 1.0), 0.03: (0.26180535615686273, 0.3464843163058824, 0.795657775117647, 1.0), 0.04: (0.27582712294117645, 0.36671691552941177, 0.812552935372549, 1.0), 0.05: (0.28527277752941177, 0.38012942263529415, 0.8234685512470589, 1.0), 0.06: (0.2994412594117647, 0.40024818329411765, 0.8398419750588235, 1.0), 0.07: (0.30906031906666664, 0.41349827226666663, 0.8501276338666667, 1.0), 0.08: (0.32371841525490197, 0.4331584405490196, 0.864722355372549, 1.0), 0.09: (0.3383765114431373, 0.45281860883137254, 0.8793170768784313, 1.0), 0.1: (0.34832334141176474, 0.4657111465098039, 0.8883461629411764, 1.0), 0.11: (0.3634607953411765, 0.4847836818509804, 0.9010188868941177, 1.0), 0.12: (0.37355243129411764, 0.4974987054117647, 0.9094673695294118, 1.0), 0.13: (0.38885187195294113, 0.5162984355764706, 0.9213734830823529, 1.0), 0.14: (0.39923148431372546, 0.5285284721568628, 0.9284591027843138, 1.0), 0.15: (0.41480090285490195, 0.5468735270274511, 0.939087532337255, 1.0), 0.16: (0.42519897019607844, 0.559058179764706, 0.9460614570784314, 1.0), 0.17: (0.4411227243607843, 0.5765318648470589, 0.9545453433843137, 1.0), 0.18: (0.4570464785254902, 0.5940055499294118, 0.963029229690196, 1.0), 0.19: (0.46767809468235294, 0.6055912316235293, 0.9685462810941176, 1.0), 0.2: (0.48385432959999997, 0.6220498496, 0.9748082026, 1.0), 0.21: (0.49463848621176465, 0.6330222615843136, 0.9789828169372549, 1.0), 0.22: (0.5108243242509803, 0.6493966148235294, 0.9850787763764707, 1.0), 0.23: (0.5216962808313725, 0.6595986063529412, 0.9877360232470589, 1.0), 0.24: (0.5380042157019607, 0.6749015936470587, 0.9917218935529412, 1.0), 0.25: (0.5543118699137254, 0.6900970112156862, 0.9955155482352941, 1.0), 0.26: (0.5651815812235294, 0.6994384449411764, 0.9966350701176471, 1.0), 0.27: (0.5814861481882353, 0.7134505955294117, 0.9983143529411764, 1.0), 0.28: (0.5923558594980393, 0.7227920292549019, 0.9994338748235294, 1.0), 0.29: (0.6085473603411764, 0.7357252298235294, 0.9993538252980392, 1.0), 0.3: (0.6193179451882354, 0.7441207347647059, 0.9989309188196078, 1.0), 0.31: (0.6354738224588236, 0.7567139921764706, 0.9982965591019608, 1.0), 0.32: (0.6461128107647058, 0.7644364965294117, 0.9968684625058823, 1.0), 0.33: (0.6619678959411764, 0.7754914668823529, 0.9939365253764706, 1.0), 0.34: (0.677822981117647, 0.786546437235294, 0.9910045882470588, 1.0), 0.35: (0.6881884831921569, 0.7931783792980391, 0.9880381043568628, 1.0), 0.36: (0.7035868880862746, 0.8025856365215686, 0.9828471328745098, 1.0), 0.37: (0.7138524913490196, 0.8088571413372548, 0.9793864852196078, 1.0), 0.38: (0.7289695795686274, 0.8174641357058824, 0.973187668372549, 1.0), 0.39: (0.7388259949411764, 0.8225716218235294, 0.9682610638235294, 1.0), 0.4: (0.753610618, 0.830232851, 0.960871157, 1.0), 0.41: (0.7633627801019607, 0.8350922218196078, 0.9556576765568627, 1.0), 0.42: (0.777377532854902, 0.8409212149490196, 0.9461493015921568, 1.0), 0.43: (0.7913922856078431, 0.8467502080784314, 0.9366409266274509, 1.0), 0.44: (0.8006008472941177, 0.8503583215607843, 0.9300075603921568, 1.0), 0.45: (0.8136925818823529, 0.8542818385490196, 0.9184801025098039, 1.0), 0.46: (0.8224204049411765, 0.8568975165411765, 0.9107951305882354, 1.0), 0.47: (0.8353447113529412, 0.8605139972941176, 0.8989704099411765, 1.0), 0.48: (0.8433581741921568, 0.8618196540156863, 0.8900171168901961, 1.0), 0.49: (0.8553783684509804, 0.8637781390980391, 0.8765871773137255, 1.0), 0.5: (0.8674276350862745, 0.864376599772549, 0.8626024620196079, 1.0), 0.51: (0.8755573874313726, 0.860242158862745, 0.8514300660980393, 1.0), 0.52: (0.8877520159490196, 0.8540404974980391, 0.8346714722156863, 1.0), 0.53: (0.8958817682941177, 0.8499060565882353, 0.8234990762941177, 1.0), 0.54: (0.9061541340352941, 0.8420910651764706, 0.8061505930823529, 1.0), 0.55: (0.9127650614705882, 0.8366818943529412, 0.7945121117647058, 1.0), 0.56: (0.9226814526235294, 0.8285681381176471, 0.7770543897882353, 1.0), 0.57: (0.9281160096666666, 0.8221971488627451, 0.765141349254902, 1.0), 0.58: (0.9357737696666666, 0.8122367012392158, 0.7471564735843139, 1.0), 0.59: (0.9434315296666667, 0.8022762536156862, 0.7291715979137255, 1.0), 0.6: (0.9473454036, 0.7946955048, 0.7169905058, 1.0), 0.61: (0.9527607176705882, 0.7829647976, 0.6986457713058823, 1.0), 0.62: (0.9563709270509804, 0.7751443261333334, 0.6864159483098039, 1.0), 0.63: (0.9605811984235294, 0.7625010185254902, 0.6679635471019607, 1.0), 0.64: (0.9627082783294117, 0.7535573465568628, 0.655601211227451, 1.0), 0.65: (0.9658988981882353, 0.7401418386039216, 0.6370577074156862, 1.0), 0.66: (0.9675442976352941, 0.7308497161882352, 0.6246854782352941, 1.0), 0.67: (0.9685329496823529, 0.7158412919058823, 0.6060967478823529, 1.0), 0.68: (0.9695216017294117, 0.7008328676235294, 0.5875080175294117, 1.0), 0.69: (0.9696829796666666, 0.6904839307372549, 0.5751383613647059, 1.0), 0.7: (0.9684997476666667, 0.673977379772549, 0.5566492560470588, 1.0), 0.71: (0.9677109263333333, 0.6629730124627451, 0.5443231858352942, 1.0), 0.72: (0.9660167198392157, 0.6461297415882352, 0.5258903482588235, 1.0), 0.73: (0.963806056435294, 0.6341884145294118, 0.5137208491529413, 1.0), 0.74: (0.9604900613294117, 0.6162764239411764, 0.49546660049411767, 1.0), 0.75: (0.9566532109764706, 0.598033822717647, 0.4773022923529412, 1.0), 0.76: (0.9530536002470588, 0.5852108672980392, 0.465372634627451, 1.0), 0.77: (0.9476541841529411, 0.5659764341686274, 0.4474781480392157, 1.0), 0.78: (0.9440545734235294, 0.5531534787490197, 0.4355484903137255, 1.0), 0.79: (0.9367796132117647, 0.5327495001098039, 0.41809333948627453, 1.0), 0.8: (0.9318312966, 0.5190855232, 0.4064796086, 1.0), 0.81: (0.9244088216823529, 0.49858955783529413, 0.38905901227058826, 1.0), 0.82: (0.9182816725843137, 0.48417347218039214, 0.37779392507058823, 1.0), 0.83: (0.908908026654902, 0.46243263716862765, 0.36095039415294133, 1.0), 0.84: (0.8995343807254902, 0.4406918021568627, 0.34410686323529416, 1.0), 0.85: (0.8921375427882353, 0.4253887370980392, 0.33328927276078435, 1.0), 0.86: (0.8808963866470588, 0.4023312782745098, 0.3171151874901961, 1.0), 0.87: (0.8734022825529412, 0.3869596390588235, 0.3063324639764706, 1.0), 0.88: (0.8610536002941176, 0.3629157635294118, 0.2906281271764706, 1.0), 0.89: (0.8523781350078431, 0.34649194649411763, 0.2803464686980392, 1.0), 0.9: (0.8393649370784314, 0.32185622094117644, 0.26492398098039216, 1.0), 0.91: (0.8301865219490197, 0.30473276355294115, 0.25489142806666665, 1.0), 0.92: (0.8155083866078432, 0.2777809871764706, 0.24029356566666665, 1.0), 0.93: (0.8008302512666666, 0.2508292108, 0.22569570326666666, 1.0), 0.94: (0.7905615319411765, 0.23139699905882352, 0.21624203829411764, 1.0), 0.95: (0.7743368501529412, 0.19975926804705882, 0.2025345544352941, 1.0), 0.96: (0.763520395627451, 0.17866744737254903, 0.1933962318627451, 1.0), 0.97: (0.7468380122117647, 0.14002101948235293, 0.17999609695686275, 1.0), 0.98: (0.7350766252941177, 0.10445963105882351, 0.17149230125490195, 1.0), 0.99: (0.717434544917647, 0.05111754842352939, 0.15873660770196077, 1.0), 1.0: (0.717434544917647, 0.05111754842352939, 0.15873660770196077, 1.0)}
        # data = xr.load_dataset('gebco_2023_n15.9082_s11.5225_w69.7148_e74.6367.nc')
        # elevation = data.elevation

        # li = np.array([[0]*COLUMNS for _ in range(ROWS)])
        # mini = float('inf')
        # maxi = float('-inf')
        # for lat in range(96):
        #     for lon in range(96):
        #         li[95-lat][lon] = (elevation[(lat)*int(data.sizes['lat']/96):(lat+1)*int(data.sizes['lat']/96), (lon)*int(data.sizes['lon']/96):(lon+1)*int(data.sizes['lon']/96)].mean().load())
        #         mini = min(mini, li[95-lat][lon])
        #         maxi = max(maxi, li[95-lat][lon])

        # print(li[-1])
        # li = (li-mini)
        # li = li/(maxi-mini)
        # print(li[-1])
        # with open("test.txt", "rb") as fp:   # Unpickling
        #     li = pickle.load(fp)

        with open('C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\ProbSims\\distributions_data.pkl', 'rb') as fp:
                distributions_data = pickle.load(fp)
        
        bathy_li = distributions_data['bathy_li']
        gauss_distr = distributions_data['gaussian']
        lat_lng_li = distributions_data['lat_lng_li']
        circular_dist = distributions_data['circular_uniform']
        shrinked_rd_dist = distributions_data['shrinked_rd_dist']
        self.lkp_latitude = distributions_data['lkp_latitude']
        self.lkp_longitude = distributions_data['lkp_longitude']
        gauss_distr = gauss_distr/gauss_distr.sum()
        circular_dist = circular_dist/circular_dist.sum()
        shrinked_rd_dist = shrinked_rd_dist/shrinked_rd_dist.sum()
        first_dist = gauss_distr*0.5 + circular_dist*0.5
        final_dist = first_dist*0.7 + shrinked_rd_dist*0.3

        self.plane_coords = [30, 40]
        self.tilemap = Tilemap(self, ROWS, COLUMNS, CELL_SIZE, self.cmap, bathy_li, final_dist, lat_lng_li, self.plane_coords)
        self.searcher = Searcher(self, SEARCHER_COLOR, CELL_SIZE, [0,0])
        self.surface = pygame.Surface((CELL_SIZE*COLUMNS, CELL_SIZE*ROWS), pygame.SRCALPHA)
        self.background_image = pygame.image.load('C:\\Users\\Manan Kher\\OneDrive\\Documents\\MINI_PROJECT\\Plane-Crash-Bayesian-Search\\Plane_S\\Djano_Pygbag_Website\\mysite\\ProbSims\\images\\gebco_2023_n4.09_s2.59_w84.15_e85.65_relief.png').convert_alpha()
        self.background_image = pygame.transform.scale(self.background_image, (CELL_SIZE*COLUMNS, CELL_SIZE*ROWS))

    async def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
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

            self.screen.blit(self.surface, (50,50))
            self.surface.blit(self.background_image, (0, 0))
            self.tilemap.render(self.surface)
            self.searcher.render(self.surface)
            pygame.draw.line(self.screen, (255, 255, 255), (45, self.screen_height-45), (self.screen_width-50, self.screen_height-45))
            pygame.draw.line(self.screen, (255, 255, 255), (45, 50), (45, self.screen_height-45))
            step = 56
            for i,x in enumerate(range(50, self.screen_width, step)):
                pygame.draw.line(self.screen, (255, 255, 255), (x, self.screen_height-45),
                                (x, self.screen_height-35))
                start_longitude = self.lkp_longitude - 0.75
                end_longitude = self.lkp_longitude + 0.75
                num_ticks = (self.screen_width-100)/step
                diff = (end_longitude - start_longitude)/num_ticks
                # font = pygame.freetype.SysFont('Sans', 13)
                # text = str(round(start_longitude+i*diff, 2))
                # text_rect = font.get_rect(text)
                # text_rect.center = (x, self.screen_height-25)
                # font.render_to(self.screen, text_rect, text, (255, 255, 255))
                font = pygame.font.Font(None, 16)
                text = str(round(start_longitude+i*diff, 2))
                text_surf = font.render(text, True, (255,255,255))
                text_rect = text_surf.get_rect(center=(x, self.screen_height-25))
                self.screen.blit(text_surf, text_rect)
            for i,y in enumerate(range(self.screen_height-50, 0, -step)):
                pygame.draw.line(self.screen, (255, 255, 255), (45, y),
                                (38, y))
                start_latitude = self.lkp_latitude - 0.75
                end_latitude = self.lkp_latitude + 0.75
                num_ticks = (self.screen_height-100)/step
                diff = (end_latitude - start_latitude)/num_ticks
                font = pygame.freetype.SysFont('Sans', 13)
                text = str(round(start_latitude+i*diff, 2))
                text_rect = font.get_rect(text)
                text_rect.center = (20, y)
                font.render_to(self.screen, text_rect, text, (255, 255, 255))


            path_to_max_prob = self.searcher.search_highest_probability_square()
            searcher_coords = self.searcher.coords.copy()
            self.searcher.update(path_to_max_prob)
            new_searcher_coords = self.searcher.coords.copy()
            if searcher_coords!=new_searcher_coords or len(path_to_max_prob)==1:
                self.tilemap.update(self.searcher)

            pygame.display.update()
            self.clock.tick(10)
            await asyncio.sleep(0)


asyncio.run(Game().run())
