# Bayesian Search Theory

# Probability that object is there (based on last known position and time it was lost(which helps us to take into consideration weather conditions))
#  Sum equals 1
grid1 = [
    [0.02, 0.02, 0.02, 0.01],
    [0.08, 0.17, 0.17, 0.01],
    [0.08, 0.17, 0.17, 0.01],
    [0.02, 0.02, 0.02, 0.01],
]

# Probability that object is found given it is there (based on depth of the ocean)
# Remains constant
grid2 = [
    [0.91, 0.97, 0.97, 0.98],
    [0.86, 0.29, 0.69, 0.99],
    [0.67, 0.65, 0.86, 0.97],
    [0.99, 0.91, 0.92, 0.97],
]

# Total probability of finding phone (this decides where to search)
# Creates an empty grid filled with zeros
totProb = [[0 for _ in range(len(grid1[0]))] for _ in range(len(grid1))]

isFound = "No"

while isFound == "No":

    for i in range(len(grid1)):
        for j in range(len(grid1[0])):
            totProb[i][j] = grid1[i][j] * grid2[i][j]
    for row in totProb:
        for value in row:
            print(format(value, '.2f')," ", end="")
        print()

    r = int(input("Enter which row to search:")) - 1
    c = int(input("Enter which column to search:")) - 1

    isFound = input("Is the object found? Type Yes or No: ")
    isFound = isFound.capitalize()

    grid1[r][c] = ((1 - grid2[r][c])*grid1[r][c]) / (1 - grid1[r][c]*grid2[r][c])
    for i in range(len(grid1)):
        for j in range(len(grid1[0])):
            if not (i==r and j==c):
                grid1[i][j] = grid1[i][j] / (1 - grid1[r][c]*grid2[r][c])

    