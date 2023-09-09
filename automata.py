import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from matplotlib import animation


# iterate over the grid and return an updated version
def iterate(grid : np.ndarray, rules : list, radius : int, defaultState : int = 1) -> np.ndarray:
    newGrid = np.zeros(grid.shape,dtype=int)

    dim = radius * 2 + 1 # cells along each axis to look at for each point
    maxArea = dim**2

    # iterate over each cell
    delta = list(range(-radius,radius+1))
    for x, y in np.ndindex(grid.shape):

        # count neighbors
        xmin = x-radius#max(x-radius, 0)
        xmax = x+radius+1#min(x+radius+1, grid.shape[0])
        ymin = y-radius#max(y-radius, 0)
        ymax = y+radius+1#min(y+radius+1, grid.shape[1])

        area = grid[xmin:xmax,ymin:ymax]
        ct = np.sum(area) + defaultState * (maxArea - area.size) - grid[x,y]

        '''
        ct = 0
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                xx = x + dx
                yy = y + dy
                if xx > 0 and yy > 0 and xx < grid.shape[0] and yy < grid.shape[1]:
                    ct += grid[xx,yy]
                else:
                    ct += defaultState
        ct -= grid[x,y]
        '''
        
        # update new grid using rules and count
        #print(x,'',y,'',ct)
        if ct in rules[grid[x,y]].keys():
            newGrid[x,y] = rules[grid[x,y]][ct]
        else:
            newGrid[x,y] = 0


    return newGrid



# grid dimensions
dim = (100,100)
# "layers" of neighbors
radius = 1
# rules
'''rules = [
    {3 : 1},
    {2 : 1, 3 : 2, 4 : 1},
    {3 : 2, 4 : 3, 5 : 2},
    {4 : 3, 5 : 3, 6 : 3},
]'''
rules = [
    {3 : 1},
    {2 : 1, 3 : 1}
]
# number of iteration cycles
nCycles = 10
## number of states
nStates = 2
# default state
defaultState = 0
animate = True

ncts = ((radius*2 + 1)**2 -1) * (nStates -1)
grid = rn.randint(low=0,high=nStates,size=dim)
#grid = np.zeros(dim,dtype=int)
#grid[40:60,40:60] = 1
#grid[45:55,45:55] = 2
#grid[49:40,49:50] = 3


# simulate and render
if animate:
    fig, ax = plt.subplots()
    im = plt.imshow((grid).astype(np.uint8))

    def animate(frame, grid : np.ndarray, rules : list, radius : int, defaultState : int = 1):
        grid[:,:] = iterate(grid,rules,radius,defaultState)
        #grid = rn.randint(low=0,high=nStates,size=dim)
        im.set_data((grid).astype(np.uint8))

    ani = animation.FuncAnimation(fig, animate, interval=500, fargs=(grid,rules,radius,defaultState))
    plt.show()
    exit(0)
else:
    for i in range(nCycles):
        grid[:,:] = iterate(grid,rules,radius,defaultState)
    fig, ax = plt.subplots()
    im = plt.imshow((grid).astype(np.uint8))
    plt.show()