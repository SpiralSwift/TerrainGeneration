# TerrainGeneration
simple, cellular-automaton-based terrain generation demos

NOTE: These have not been properly documented, yet; and many scripts contain redundant functions

automata.py: a basic cellular automaton simulation with flexible rulesets.
Rules, interaction radii, and densities can be set in the script.
Currently set up to run Conway's Game of Life.

caves.py: very simple procedural cave generator.
Creates reasonably believable 2D caves using competing cellular automata.
Yellow cells indicate walls, and purple cells indicate open floor.
Walls are generated along the boundaries of the grid.

competition.py: example of competing cellular automata with variable numbers of competing populations.

dungeon.py: procedural dungeon generator ported from an older project.
Creates sensibly-proportioned rectangular rooms with randomized "difficulty" levels.
The script determines the "easiest" path between the start and end rooms on positioned opposite edges of the grid, then connects isolated rooms randomly.
Walls are indicated in black, and doors are represented as gaps in walls between colored cells (rooms). Colors indicate "difficulty."
NOTE: this script is ancient and badly in need of an update and proper documentation.
