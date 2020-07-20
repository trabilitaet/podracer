# podracer

Codebase for semester thesis on controlling a hovercraft style simulated racing bot. See [coders strike back website](https://www.codingame.com/ide/puzzle/coders-strike-back)

# contents
## Referee.java
Reference judge implementation provided by codeingame developers.

## pygamecsb folder
Python implementation of the game, dynamics copied from judge but providing better handling (logging, installation of packages) than using webinterface.

### pod.py
Contains dynamics and rendering of bot. Called from csb.py to provide this implementation. Do not change when trying to implement a controller.

### csb.py
Provides the main runtime as well as the interface between the controller and the environment. Call controller code in the section marked. 
Available measurements/ outputs: position (x,y), checkpoint coordinates (for all checkpoints). Additional data includes velocity and heading.
Provide desired target heading and thrust.