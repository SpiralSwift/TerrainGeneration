import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rn


# ----- CLASSES -----

class world:
    width = 10
    height = 10
    rescale = 5
    cwidth = 10
    cheight = 10
    ndxns = 4
    
    nseeds = 15
    cindex = 0
    crank = 0
    
    iterations = 20
    nslices = 5
    reiterations = 5
    
    limit = 3
    ratio = 2
    
    nranks = 5


class room:
    def __init__(self,vtx):
        # identifier
        self.index = world.cindex
        world.cindex += 1
        
        # ranking
        self.rank = world.crank +3
        world.crank += 1
        if world.crank == world.nranks: world.crank = 0
        
        # expansion logic
        self.vtx = vtx # [[x1, x2], [y1, y2]]
        self.dim = np.ones(2) # [width, height]
        self.blocked = np.array([False,False,False,False]) # [right, up, left, down]
        
        # connecticity logic
        self.subRank = np.zeros(3) # [steps from start, steps from end, steps from test room]
        self.tempRank = 0
        self.neighbors = []
        self.neighborDxns = []
        self.connections = []
        
        return
    
    
    def update_dim(self):
        self.dim = np.array([self.vtx[0][1]-self.vtx[0][0],self.vtx[1][1]-self.vtx[1][0]])
        return
    
    
    def rescale(self):
        self.vtx *= world.rescale
    
    
    def expand(self,dxn,rooms):
        if self.blocked[dxn] : return False
        if not self.check_blocked(dxn,rooms):
            if dxn == 0 : self.vtx[0][1] += 1
            elif dxn == 1 : self.vtx[1][1] += 1
            elif dxn == 2 : self.vtx[0][0] -= 1
            elif dxn == 3 : self.vtx[1][0] -= 1
            self.update_dim()
            return True
        
        
    def check_and_slice(self,limboRooms):
        # check if exceeds size limit
        if self.dim[0] <= world.limit and self.dim[1] <= world.limit : return False
        
        # check if dimension ratios exceed limit
        if self.dim[0] / self.dim[1] >= world.ratio:
            nx = self.vtx[0][0] + self.dim[0]/2 + rn.randint(3) -1
            
            nvtx = np.copy(self.vtx)
            nvtx[0][0] = nx
            limboRooms.append(room(nvtx))
            
            self.vtx[0][1] = nx
            self.update_dim()
            return True
        
        elif self.dim[1] / self.dim[0] >= world.ratio:
            ny = self.vtx[1][0] + self.dim[1]/2 + rn.randint(3) -1
            
            nvtx = np.copy(self.vtx)
            nvtx[1][0] = ny
            limboRooms.append(room(nvtx))
            
            self.vtx[1][1] = ny
            self.update_dim()
            return True
        
        return False
            
    
    def check_blocked(self,dxn,rooms):
        # check edges
        blocked = self.touches_edge(dxn)
        
        # check for adjacent rooms
        if not blocked:
            for aroom in rooms:
                if self.is_adjascent(dxn,aroom):
                    blocked = True
                    break
        
        self.blocked[dxn] = blocked
        return blocked
    
    
    def touches_edge(self,dxn):
        if dxn == 0: # right
            return self.vtx[0][1] >= world.cwidth
        elif dxn == 1: # up
            return self.vtx[1][1] >= world.cheight
        elif dxn == 2: # left
            return self.vtx[0][0] <= 0
        elif dxn == 3: # down
            return self.vtx[1][0] <= 0
        return False
    
    
    def touches_an_edge(self):
        for dxn in range(world.ndxns):
            if(self.touches_edge(dxn)): return True
        return False
    
    
    def is_adjascent(self,dxn,aroom):
        # check not same room
        if self.index == aroom.index : return False
        
        if dxn == 0: # right
            return (self.vtx[0][1] == aroom.vtx[0][0] 
                and self.vtx[1][1] > aroom.vtx[1][0] 
                and self.vtx[1][0] < aroom.vtx[1][1])
        elif dxn == 1: # up
            return (self.vtx[1][1] == aroom.vtx[1][0] 
                and self.vtx[0][1] > aroom.vtx[0][0] 
                and self.vtx[0][0] < aroom.vtx[0][1])
        elif dxn == 2: # left
            return (self.vtx[0][0] == aroom.vtx[0][1] 
                and self.vtx[1][1] > aroom.vtx[1][0] 
                and self.vtx[1][0] < aroom.vtx[1][1])
        elif dxn == 3: # down
            return (self.vtx[1][0] == aroom.vtx[1][1] 
                and self.vtx[0][1] > aroom.vtx[0][0] 
                and self.vtx[0][0] < aroom.vtx[0][1])
        return False
    
    
    def find_neighbors(self,rooms):
        for dxn in range(world.ndxns):
            for aroom in rooms:
                if self.is_adjascent(dxn,aroom):
                    self.neighbors.append(aroom)
                    self.neighborDxns.append(dxn)
        return
    
    
    def connect_to(self,i):
        aroom = self.neighbors[i]
        dxn = self.neighborDxns[i]
        
        if(aroom in self.connections): return False, np.zeros(2)
        
        self.connections.append(aroom)
        aroom.connections.append(self)
        door = locate_door(self,aroom,dxn)
        
        return True, door
    

# ----- FUNCTIONS -----

def plant_seed(rooms,x,y):
    newRoom = room(np.array([[x,x+1],[y,y+1]]))
    rooms.append(newRoom)
    return


def draw_dungeon(rooms,grid,reset):
    if reset : rects_to_grid(rooms,grid)
    plt.figure()
    plt.imshow(grid,cmap='magma')
    plt.show()
    return
    
    
def rects_to_grid(rooms,grid):
    # reset grid
    for x in range(len(grid[0])):
        for y in range(len(grid[1])):
            grid[x][y] = 0
    
    # add rooms
    index = 5
    for r in rooms:
        for x in range(r.vtx[0][0],r.vtx[0][1]):
            for y in range(r.vtx[1][0],r.vtx[1][1]):
                grid[x][y] = index
        index += 1
    return


def locate_door(r1,r2,dxn):
    door = np.zeros(2)
    
    if dxn == 0 : door[0] = r1.vtx[0][1] -1
    elif dxn == 1 : door[1] = r1.vtx[1][1] -1
    elif dxn == 2 : door[0] = r1.vtx[0][0] -1
    elif dxn == 3 : door[1] = r1.vtx[1][0] -1
        
    if dxn == 0 or dxn == 2:
        y1 = r1.vtx[1][0] if r1.vtx[1][0] > r2.vtx[1][0] else r2.vtx[1][0]
        y2 = r1.vtx[1][1] if r1.vtx[1][1] < r2.vtx[1][1] else r2.vtx[1][1]
        dy = 1 if y1 == 0 else 0
        door[1] = rn.randint(y2-y1-1-dy) + y1+dy
    elif dxn == 1 or dxn == 3:
        x1 = r1.vtx[0][0] if r1.vtx[0][0] > r2.vtx[0][0] else r2.vtx[0][0]
        x2 = r1.vtx[0][1] if r1.vtx[0][1] < r2.vtx[0][1] else r2.vtx[0][1]
        dx = 1 if x1 == 0 else 0
        door[0] = rn.randint(x2-x1-1-dx) + x1+dx
       
    return door


def list_edge_rooms(rooms,cdxn=-1):
    rlist = []
    dxnSet = [cdxn] if cdxn != -1 else range(world.ndxns)
    
    for dxn in dxnSet:
        for r in rooms:
            if r.touches_edge(dxn) and not r in rlist: rlist.append(r)
    
    return rlist


def set_room_distances(rooms,base,index,viaConnections=False):
    rlist = rooms.copy()
    rlist.remove(base)
    templist = []
    templist.append(base)
    iteration = 0
    while len(rlist) > 0:
        iteration += 1
        #print(iteration,':',len(rlist))
        newlist = []
        for r in templist:
            for rr in (r.neighbors if not viaConnections else r.connections):
                if rr in rlist:
                    rlist.remove(rr)
                    newlist.append(rr)
                    rr.subRank[index] = iteration
        templist = np.copy(newlist)
    return


def get_linked_rooms(rooms, refRoom):
    currentRooms = [refRoom]
    linkedRooms = [refRoom]
    while len(currentRooms) > 0:
        nextRooms = []
        for r in currentRooms:
            for rc in r.connections:
                if rc not in linkedRooms:
                    linkedRooms.append(rc)
                    nextRooms.append(rc)
        currentRooms = nextRooms.copy()
            
    return linkedRooms, [r for r in rooms if r not in linkedRooms]


def connect_rooms(r1,r2,grid):
    index = r1.neighbors.index(r2)
    query, door = r1.connect_to(index)
    if query : grid[int(door[0])][int(door[1])] = r1.rank +2
    return
        
        
def connect_to_furthest(rooms,grid,base):
    # find distances of each room from base via connections
    set_room_distances(rooms, base, 2, viaConnections=True)

    # connect to furthest neighbor
    maxDist = 0
    nroom = base
    for r in base.neighbors:
        if (r.subRank[2] > maxDist) or (r.subRank[2] == maxDist and rn.randint(2) == 0):
            maxDist = r.subRank[2]
            nroom = r
    connect_rooms(base,nroom,grid)
    return


# initialize
def till_soil():
    world.cwidth = world.width
    world.cheight = world.height
    rooms = []
    minigrid = np.zeros(([world.cwidth,world.cheight]))
    return rooms, minigrid


def plant_seeds(rooms,grid):
    coords = list(range(len(grid[0])*len(grid[1])))
    
    for i in range(world.nseeds):    
        c = rn.choice(coords)
        coords.remove(c)
        
        y = int(c / len(grid[0]))
        x = int(c - y*len(grid[0]))
        
        grid[x][y] = world.cindex
        plant_seed(rooms,x,y)
    return


def grow_seeds(rooms):
    dxn = 0
    for i in range(world.iterations*world.ndxns):
        dxn += 1
        if dxn == world.ndxns : dxn = 0

        for r in rooms : r.expand(dxn,rooms)
    return


def replant_seeds(rooms,grid):
    rects_to_grid(rooms,grid)
    
    for x in range(len(grid[0])):
        for y in range(len(grid[1])):
            if grid[x][y] == 0 : plant_seed(rooms,x,y)
    return


def slice_stalks(rooms):
    for i in range(world.nslices):
        limboRooms = []
        for r in rooms : r.check_and_slice(limboRooms)
        for r in limboRooms : rooms.append(r)
    return


# rescale rooms
def grow_plants(rooms):
    for r in rooms : r.rescale()
    world.cwidth = world.width*world.rescale
    world.cheight = world.height*world.rescale
    return np.zeros(([world.cwidth,world.cheight]))


def add_walls(rooms,grid):
    valW = 0
    valF = 2
    
    rects_to_grid(rooms,grid)
    ogrid = np.copy(grid)
    
    # reset grid
    for x in range(len(grid[0])):
        for y in range(len(grid[1])):
            grid[x][y] = 0
    
    # add walls on room boundaries ('negative' side only)
    for r in rooms:
        for x in range(r.vtx[0][0],r.vtx[0][1]-1):
            for y in range(r.vtx[1][0],r.vtx[1][1]-1):
                grid[x][y] = r.rank+valF
    
    '''
    for x in range(len(grid[0])-1):
        for y in range(len(grid[1])-1):
            if ogrid[x][y] != ogrid[x+1][y] or ogrid[x][y] != ogrid[x][y+1] : grid[x][y] = valW
    '''
    
    # add walls on edges
    for x in range(len(grid[0])):
        grid[x][0] = valW
        grid[x][len(grid[1])-1] = valW
    for y in range(len(grid[1])):
        grid[0][y] = valW
        grid[len(grid[0])-1][y] = valW
    return


def connect_rooms_LEGACY(rooms,grid):
    valD = 2
    
    # locate doors
    for r in rooms:
        r.find_neighbors(rooms)
        for i in range(len(r.neighbors)):
            query, door = r.connect_to(i)
            if query : grid[int(door[0])][int(door[1])] = r.rank+valD
    return


def rank_rooms(rooms,grid):
    # find room neighbors
    for r in rooms: r.find_neighbors(rooms)
    
    # find entrance
    entrance = rn.choice(list_edge_rooms(rooms))
    entrance.rank = 0
    print('   Entrance located')
    
    # find steps from entrance
    set_room_distances(rooms,entrance,0)
    print('   Steps from entrance computed')
    
    # find exit
    rlist = list_edge_rooms(rooms)
    rlist.remove(entrance)
    maxVal = 0
    exit = entrance
    for r in rlist:
        query = (r.subRank[0] > maxVal) or (r.subRank[0] == maxVal and rn.randint(2) == 0)
        if query:
            maxVal = r.subRank[0]
            exit = r
    exit.rank = 0
    print('   Exit located')
    
    # find steps from exit
    set_room_distances(rooms,exit,1)
    print('   Steps from exit computed')
    
    # find avg rank of neighboring rooms and effective total rank
    for r in rooms:
        avgRank = r.rank
        adjRank = 0
        for rr in r.neighbors:
            avgRank += rr.rank
            if rr.rank < r.rank: adjRank += 1
        avgRank /= 1 + len(r.neighbors)
        r.tempRank = (r.rank*2 + r.subRank[1])*5 + avgRank
    entrance.tempRank, exit.tempRank = 0, 0
        
    
    return entrance, exit


# connect main sequence of rooms
def connect_rooms_1(entrance,exit,rooms,grid):
    
    rlist = [r for r in rooms if r != entrance]
    rseq = [entrance]
    stepct = 0
    atExit = False
    while not atExit:
        if stepct > len(rooms):
            print('   Error: pathfinding failed')
            return [r for r in rooms if r not in rseq], [entrance]
        print('   Step',stepct,':',rseq[-1]==entrance)
        stepct += 1
        
        # backtrack one step if hit dead end
        rcount = 0
        for r in rseq[-1].neighbors:
            if r in rlist: rcount += 1
        if rcount == 0 :
            rseq = rseq[:-1]
            print('   backtracking')
        
        # find lowest-ranked neighbor
        nroom = rseq[-1]
        minVal = 1000
        for r in rseq[-1].neighbors:
            if r in rlist:
                query = False
                if r.tempRank < minVal: query = True
                elif r.tempRank == minVal and rn.randint(2) == 0: query = True
                if query:
                    minVal = r.tempRank
                    nroom = r
        rseq.append(nroom)
        if nroom in rlist : rlist.remove(nroom)
        if nroom == exit : atExit = True
    
    # connect rooms
    for i in range(len(rseq)-1):
        connect_rooms(rseq[i],rseq[i+1],grid)
            
    return rseq


# connect all rooms to main sequence
def connect_rooms_2(rooms,grid,entrance):
    
    # add a single link to all unlinked rooms
    linkedRooms, unlinkedRooms = get_linked_rooms(rooms,entrance)
    for r in unlinkedRooms:
        rlist = [rr for rr in r.neighbors if rr not in r.connections]
        if(len(rlist)>0):
            r2 = rn.choice(rlist)
            connect_rooms(r,r2,grid)
    
    # add random connections from unlinked rooms until all rooms linked
    validRooms = unlinkedRooms.copy()
    while len(validRooms) > 0:
        linkedRooms, unlinkedRooms = get_linked_rooms(rooms,entrance)
        validRooms = [rr for rr in validRooms if rr not in linkedRooms]
        
        if len(validRooms) == 0:
            if len(unlinkedRooms) != 0: print('Error: room failed to connect')
            break
        
        r = rn.choice(validRooms)
        rlist = [rr for rr in r.neighbors if rr not in r.connections]
        if len(rlist) > 0:
            r2 = rn.choice(rlist)
            connect_rooms(r,r2,grid)
        else:
            validRooms.remove(r)
            
    return


# remove dead-ends and linear sequences
def connect_rooms_3(rooms,grid,entrance,exit):
    # remove dead ends
    while True:
        # find dead ends (ignore entrance, exit)
        rlist = [r for r in rooms if (len(r.connections) <= 1 and r != entrance and r != exit)]
        if len(rlist) == 0: break
        
        # select dead end, connect to furthest neighbor
        aroom = rn.choice(rlist)
        connect_to_furthest(rooms,grid,aroom)
    
    # remove long linear sequences
    while True:
        # find rooms in linear sequences
        rlist = []
        for r in rooms:
            query = len(r.connections) == 2
            for rr in r.connections:
                if query and len(rr.connections) != 2: query = False
            if query and len(r.neighbors) > 2: rlist.append(r)
        if len(rlist) == 0: break
        
        # select dead end, connect to furthest neighbor
        aroom = rn.choice(rlist)
        connect_to_furthest(rooms,grid,aroom)
    
    return


# ----- SIMULATION -----

rooms, minigrid = till_soil()
print('Soil Tilled')

plant_seeds(rooms,minigrid)
print('Seeds Planted')
#draw_dungeon(rooms,minigrid, True)

grow_seeds(rooms)
print('Seeds Grown')
#draw_dungeon(rooms,minigrid, True)

replant_seeds(rooms,minigrid)
print('Seeds Replanted')
#draw_dungeon(rooms,minigrid, True)

slice_stalks(rooms)
print('Stalks Sliced')
#draw_dungeon(rooms,minigrid, True)

grid = grow_plants(rooms)
#print('Plants Magnified')
#draw_dungeon(rooms,grid, True)

entrance, exit = rank_rooms(rooms,grid)
print('Roots Sprouted')

add_walls(rooms,grid)
print('Trunks Lignified')
draw_dungeon(rooms,grid, False)

rseq = connect_rooms_1(entrance,exit,rooms,grid)
print('Mycorhizoidial Networks Established (1/3)')
draw_dungeon(rooms,grid, False)

connect_rooms_2(rooms,grid,entrance)
print('Mycorhizoidial Networks Established (2/3)')
draw_dungeon(rooms,grid, False)

connect_rooms_3(rooms,grid,entrance,exit)
print('Mycorhizoidial Networks Established (3/3)')
draw_dungeon(rooms,grid, False)