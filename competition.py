import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from matplotlib import animation


# iterate over the grid and return an updated version
def iterate(grid : np.ndarray, rules : list, radius : int, nStates : int, competition : bool, defaultState : int = 0) -> np.ndarray:
    newGrid = np.zeros(grid.shape,dtype=int)

    dim = radius * 2 + 1 # cells along each axis to look at for each point
    #maxArea = dim**2

    # iterate over each cell
    for x, y in np.ndindex(grid.shape):

        # find neighbors
        xmin = x-radius#max(x-radius, 0)
        xmax = x+radius+1#min(x+radius+1, grid.shape[0])
        ymin = y-radius#max(y-radius, 0)
        ymax = y+radius+1#min(y+radius+1, grid.shape[1])
        area = grid[xmin:xmax,ymin:ymax]
        #deltaArea = maxArea - area.size

        # count neighbors of each type
        val = grid[x,y]
        if val == 0:
            # randomly choose between most-abundant states
            values, counts = np.unique(area, return_counts=True)
            if len(values) > 1:
                values = values[1:]
                counts = counts[1:]
                maxct = max(counts)
                vals = [values[i] for i in range(len(counts)) if counts[i] == maxct]
                val = rn.choice(vals)
        ct = np.count_nonzero(area == val)
        ct -= 1 * (grid[x,y] == val) # discount central cell
        #if val == defaultState : ct += deltaArea

        # factor in competitors
        if competition and val != 0:
            for i in range(1,nStates):
                if i != val:
                    ct -= np.count_nonzero(area == i)
            #if val != defaultState and defaultState != 0:
            #    ct -= deltaArea
        
        # update new grid using rules and count
        if ct in rules[grid[x,y]]:
            newGrid[x,y] = val
        else:
            newGrid[x,y] = 0
        #if newGrid[x,y] != 0 : print(grid[x,y],'',val,'',ct,'',newGrid[x,y])

    return newGrid


if __name__ == "__main__":
    dim = (100,100) # grid dimensions
    radius = 1 # "layers" of neighbors
    rules = [
        [3,4,5,6,7,8],
        [3,4,5,6,7,8],
        [2,3,4,5,6,7,8],
        [2,3,4,5,6,7,8],
    ]
    nCycles = 10 # number of iteration cycles
    nStates = 3 # number of states
    competition = True # competition flag
    animate = True # boundary value

    ncts = ((radius*2 + 1)**2 -1) * (nStates -1)
    grid = rn.randint(low=0,high=nStates,size=dim)


    # simulate and render
    if animate:
        fig, ax = plt.subplots()
        im = plt.imshow((grid).astype(np.uint8))

        def animate(frame, grid : np.ndarray, rules : list, radius : int, nStates : int, competition : bool, defaultVal : int = 0):
            grid[:,:] = iterate(grid,rules,radius,nStates,competition,defaultVal)
            #grid = rn.randint(low=0,high=nStates,size=dim)
            im.set_data((grid).astype(np.uint8))

        ani = animation.FuncAnimation(fig, animate, interval=50, fargs=(grid,rules,radius,nStates,competition))
        plt.show()
        exit(0)
    else:
        for i in range(nCycles):
            grid[:,:] = iterate(grid,rules,radius,nStates,competition)
        fig, ax = plt.subplots()
        im = plt.imshow((grid).astype(np.uint8))
        plt.show()