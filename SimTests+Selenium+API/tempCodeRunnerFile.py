data)
# elevation = data.elevation
# # print(elevation[1][0].load)
# # print(elevation[:int(data.dims['lat']/75), :int(data.dims['lon']/84)])
# li = np.array([[0]*84 for _ in range(75)])
# mini = float('inf')
# maxi = float('-inf')
# for lat in range(75):
#     for lon in range(84):
#         li[74-lat][lon] = (elevation[(lat)*int(data.dims['lat']/75):(lat+1)*int(data.dims['lat']/75), (lon)*int(data.dims['lon']/84):(lon+1)*int(data.dims['lon']/84)].mean().load())
#         mini = min(mini, li[74-lat][lon])
#         maxi = max(maxi, li[74-lat][lon])
# print(li)
# print(mini)
# print(maxi)
# # norm = [[(float(j)-mini)/(maxi-mini) for j in range(len(li[i]))] for i in range(len(li))]
# # print(norm[-1][1])
# # norm = [(float(i)-min(li))/(max(li)-min(li)) for i in li]
# # normalized_arr = preprocessing.normalize(li)
# # print(normalized_arr)
# # for i in range(len(li)):
# #     for j in range(len(li[i])):
# #         li[i][j] += abs(mini)
# # scaler = preprocessing.MinMaxScaler()
# # d = scaler.fit_transform(li)
# # print(d[-1])
# # print(li[-1])
# # sns.heatmap(d)
# # plt.plot()

# li = li+abs(mini)
# print(li)