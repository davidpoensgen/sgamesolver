# sgamesolver  
  
**Stochastic game solver**, short: sgamesolver, is a python package to compute 
stationary or Markov-perfect equilibria of stochastic games, using the 
homotopy continuation method.  

Some useful links:
- Source code is hosted on [github](https://github.com/davidpoensgen/sgamesolver)
- Also on github is an [issue tracker](https://github.com/davidpoensgen/sgamesolver/issues):, where you are welcome to submit bug reports or feature requests
- Documentation is found on [readthedocs](https://sgamesolver.readthedocs.io/en/latest/)

sgamesolver is free and open source, under the MIT license. 
If you use the program for any published research, please cite 
[Eibelshäuser/Poensgen (2019)](https://dx.doi.org/10.2139/ssrn.3316631).

## Installation
sgamesolver is hosted on PyPI, so installation is usually as simple as
 ````sh
pip install sgamesolver
````
Head over to the docs for other options for installation.

## Usage
Solving a stochastic game is done in three steps: 
- define a game
- pick a homotopy function
- set up and run the solver

````python
import sgamesolver

game = sgamesolver.SGame.random_game(64, 2, 4, seed=42)

qre = sgamesolver.homotopy.QRE(game)

qre.solver_setup()
qre.solve()

print(qre.equilibrium)
````


A quick rundown of these steps – more details are found in the docs:

##### 1. Set up a stochastic game
````python
game = sgamesolver.SGame.random_game(64, 2, 4, seed=42)
````

Stochastic games are represented by the class `SGame`. For this quick example, we are using the method `random_game` to randomize a game with 64 states, 2 players, and 4 actions per player and state. (Setting a seed just makes the result reproducible.) 

Of course, almost universally you'll want to solve a specific game, 
rather than a random one. The 
[documentation](https://sgamesolver.readthedocs.io/en/latest/) contains instructions 
and examples on how to create an `SGame` which represents the specific 
game you want to solve.

##### 2. Select and set up a homotopy function for your game
```python
qre = sgamesolver.homotopy.QRE(game)
```

sgamesolver uses the **homotopy principle** to solve stochastic games, 
a general technique to solve systems of non-linear equations. 
In short, the idea is as follows: Instead of solving some very hard problem directly
(in our case: finding an equilibrium), a continuous transformation 
is applied to the system, yielding a related, but much simpler problem, 
for which one can easily obtain a solution. This transformation is then 
gradually reversed while tracking the solution, until arriving at a solution 
for the original problem – here, the desired stationary equilibrium. 
(You can find more background in the documentation – 
although such knowledge is not necessary for using the program.)

The (mathematical) function used for this transformation is called 
**homotopy function**. In general, there are many possibilities 
to construct a suitable one. sgamesolver currently includes two: 
The one we picked for this example, `sgamesolver.homotopy.QRE`, is on 
an extension of quantal response equilibrium to stochastic games 
 [(Eibelshäuser/Poensgen 2019b)](https://dx.doi.org/10.2139/ssrn.3314404). The other, `sgamesolver.homotopy.LogTracing`, implements the 
logarithmic tracing procedure for stochastic games [(Eibelshäuser/Klockmann/Poensgen/von Schenk 2022)](https://dx.doi.org/10.2139/ssrn.3748830). 
Which one to pick? In our experience, the former is more robust – 
while the latter has the advantage that it allows to search for multiple equilibria. 
More homotopy functions are to come! In any case, please makle sure to also cite the paper that 
introduced the homotopy you end up using.

##### 3. Let the homotopy solver do its job
Finally, we will set up the solver and start it:
```python
qre.solver_setup()
qre.solve()
```

Then it's time to lean back and watch for a bit:
```
==================================================
Start homotopy continuation
Step    37: t =  3.612 ↑, s =  20.47, ds =  3.418
```
... until ...
```
Step   247: t = 1.147e+04 ↑, s = 9.385e+04, ds =  1000.
Step   247: Continuation successful. Total time elapsed: 0:00:15
End homotopy continuation
==================================================
An equilibrium was found via homotopy continuation.
```
... success!
Ideally, the solver will be able to find a solution without
requiring any further interaction, as in this example. In cases 
where this does not work out, check out the 
[section on troubleshooting](https://sgamesolver.readthedocs.io/en/latest/troubleshooting.html) 
in the documentation.

##### 4. Aftermath
We can now display the solution:
```python
print(qre.equilibrium)
```
which outputs equilibrium strategies and values for all 64 states:
```
+++++++++ state00 +++++++++
                       a0    a1    a2    a3  
player0 : v=15.09, σ=[1.000 0.000 0.000 0.000]
player1 : v=15.63, σ=[0.000 0.000 1.000 0.000]

+++++++++ state01 +++++++++
                       a0    a1    a2    a3  
player0 : v=14.76, σ=[0.000 0.961 0.000 0.039]
player1 : v=15.61, σ=[0.354 0.000 0.000 0.646]

+++++++++ state02 +++++++++
                       a0    a1    a2    a3  
player0 : v=14.84, σ=[1.000 0.000 0.000 0.000]
player1 : v=15.61, σ=[0.000 0.000 1.000 0.000]

... (abridged here for brevity) ...

+++++++++ state63 +++++++++
                       a0    a1    a2    a3  
player0 : v=14.92, σ=[0.000 1.000 0.000 0.000]
player1 : v=15.75, σ=[1.000 0.000 0.000 0.000]
```
Of course, you now also can access equilibrium strategies (and values) 
as `numpy` arrays and use them for further calculations or simulations.
```
eq_strat = qre.equilibrium.strategies
eq_values = qre.equilibrium.values
```


