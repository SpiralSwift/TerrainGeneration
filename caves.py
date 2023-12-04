import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from matplotlib import animation


def iterate(grid : np.ndarray, rules : np.ndarray, radius : int, defaultState : int = 1) -> np.ndarray:
    """
    iterate over the grid and return an updated version
    """
    newGrid = np.zeros(grid.shape,dtype=int)

    dim = radius * 2 + 1 # cells along each axis to look at for each point
    nNeighbors = rules.shape[1]-2
    nStates = rules.shape[0]

    # iterate over each cell
    delta = list(range(-radius,radius+1))
    for x, y in np.ndindex(grid.shape):

        # icount neighbors
        xmin = max(x-radius, 0)
        xmax = min(x+radius+1, grid.shape[0])
        ymin = max(y-radius, 0)
        ymax = min(y+radius+1, grid.shape[1])

        area = grid[xmin:xmax,ymin:ymax]
        ct = np.sum(area) + defaultState * (nNeighbors - area.size +1) - grid[x,y]
        
        # update new grid using rules and count
        #print(x,'',y,'',ct)
        newGrid[x,y] = rules[grid[x,y],ct]


    return newGrid


def run_automaton(grid : np.ndarray, cts : list, radius : int, nCycles : int, defaultState : int = 1, nStates : int = 2) -> np.ndarray:
    rules = get_rules(cts,radius,nStates)
    newGrid = np.zeros(grid.shape,dtype=int)
    newGrid[:,:] = grid

    for i in range(nCycles):
        newGrid[:,:] = iterate(newGrid,rules,radius,defaultState)
    return newGrid


def get_rules(cts, radius : int, nStates : int = 2) -> np.ndarray:
    nNeighbors = (radius*2 + 1)**2 -1
    rules = list([[1 if i >= cts[j][0] and i <= cts[j][1] else 0 for i in range(nNeighbors+2)] for j in range(nStates)])
    return np.asarray(rules)


def plot_grid(plt,grid) -> None:
    plt.imshow((grid).astype(np.uint8))
    plt.show()


if __name__ == "__main__":
    # setup
    dim = (200,200)
    nStates = 2
    grid = rn.randint(low=0,high=nStates,size=dim)

    fig, ax = plt.subplots()

    # rough tunnel pattern
    cts = [[7,9],[6,10]]
    grid[:,:] = run_automaton(grid,cts,2,3)
    #plot_grid(plt,grid)

    # clean
    cts = [[4,9],[3,9]]
    grid[:,:] = run_automaton(grid,cts,1,5)
    plot_grid(plt,grid)