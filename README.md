# podracer

Codebase for semester thesis on controlling a hovercraft style simulated racing bot. See [coders strike back website](https://www.codingame.com/ide/puzzle/coders-strike-back)

# how to use
1. check out branch of controller to try (e.g. git checkout controller_A)
2. run the demo from a commandline or an IDE: python csb.py
3. look at commandline outputs, animation or logfiles for results

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

### controller_ .py
Contains an implementation of a controller. Called from csb.py