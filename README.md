# podracer

Codebase for semester thesis on controlling a hovercraft style simulated racing bot. See [coders strike back website](https://www.codingame.com/ide/puzzle/coders-strike-back)

# how to use
1. ~~check out branch of controller to try (e.g. git checkout controller_A)~~
1. check out master
2. change controller call in csb.py to the controller under test
3. run the demo from a commandline or an IDE: python csb.py
4. look at commandline outputs, animation or logfiles for results
5. to run mutliple times with different seeds, use run.py

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
##test scripts folder
Contains scripts to test various utility function, such as jacobian or hessian.

#results
Running the simulation results in a score file being generated containing the number of steps taken to reach the target.