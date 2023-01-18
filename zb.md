
# Shadok

This is my personal python package, that I created mostly for learning purposes, but it contains some usefull
tools like wrappers, dictionnaries (in French), a class that helps you print a progress bar, tools for memorizing variables, solvers for the french game "Des Chiffres et des Lettres" and Scrabble...



## Installation

Install shadok where your Python Libraries are with :

```bash
  git clone https://github.com/ProfesseurShadoko/shadok.git
```

## Documentation

***Style***<br>
Print colored text : 
```python
from shadok.style import style
style = Style(style="underline",text="red",background="white")
print(style("Some text"))
```
***
***Wrappers***<br>
Implement Fibonacci :
```python
from shadok.wrappers import memoize_me

@memoize_me
def fib(n):
  if n in [0,1]:
    return 1
  else:
    return fib(n-1)+fib(n-2)
```

Time any function :
```python
from shadok.wrappers import time_me

@time_me
def test():
  return fib(30)
  
```

Intercept everything a function has printed :
```python
from shadok.wrappers import jam_me, JAMMER

@jam_me
def prints_stuff():
  print("stuff")
  
print_stuff() #doesn't print anything
JAMMER.print() #prints stuff
  
```

***

***Progress Bar***

Trough an Iteration :
```python
from shadok.progress_bar import ProgressIterator
from time import sleep

for i in ProgressIterator(range(0,100)):
  print(i)
  sleep(0.1)
```

Trough a wrapper:
```python
from shadok.progress_bar import ProgressBar, task_me

bar = ProgressBar(100)

@task_me(bar,1)
def wait():
  sleep(0.1)

for i in range(100):
  wait()  
```

***

***Dictionnary***
```python
from shadok.dictionnary import Wordle
from shadok.style import OK,FAIL

dico = Wordle()
word = "apple"

if dico[word]: #check weather the dictionnary contains a word
  OK()
else:
  FAIL()
```

***
***XLDB***

Implementation of a Database and Model via an Excel Sheet (requires openpyxl, script tries to install it automatically at start)

```python
from shadok.XLDB import Database,Model,Filter

class Admin(Model):
  def __init__(self,name:str):
    super().__init__()
    self.name = name
    
  @staticmethod
  def get_property_names() -> list:
    return ["name"]
 
Database.start()
Admin.create()

Admin("Professeur Shadoko").save()
print(Admin.get(1))
print(Admin.filter(Filter(lambda x:x.id<12,name="Professeur Shadoko"))[0])
```

***
***Des Chiffres et Des Lettres (French TV show)***

Letters : the rules are the following. You are given a set of 10 letters, you must created the longest word possible with it.
```python
from shadok.dcdl import LetterSolver
from shadok.dictionnary import Dcdl #official dictionnary of the games (some words are not allowed, conjugated verbs for exemple)

LetterSolver("ceolclcnic",Dcdl()).run() #returns 'coccinelle'
```

Numbers : the rules are the following. You are given a set of 6 numbers (integers from 1 to 10 plus 25,50,75 and 100) and using + - * / to combine the numbers you must find some target number
```python
from shadok.dcdl import ChiffreSolver

ChiffreSolver([1,25,75,4,3,6],985).run() #returns (75-1+6)*4*3+25=985
```
***

***Shell***

Very simplistic implementation of a shell that allows you to check if a word exists or to solve problems from *Des Chiffres et Des Lettres*
```python
import shadok.shell
```
***

***Memory***

An easy way to save and load variable in python, using the pickle package.
```python
from shadok.memory import Memory

class Pet:
  def __init__(self,name:str):
    self.name = name

dog = Pet("Snoopy")
Memory().save(dog,"dog")

dog = Memory().load("dog")
print(dog.name)
```

It is also possible to create a class that, when an instance is constructed, first checks if an instance already exists in the memory (if so, it will be loaded, else,  the usual __new__ and __init__ function will be called.
```python
from shadok.memory import Loadable

class Dog(Loadable):
  def __init__(self,name:str):
    self.name = name

dog = Dog("Snoopy")
dog.save()

dog = Dog()
print(dog.name) #prints 'Snoopy'
```
***



***

And that's pretty much it ! For now at least 😏

    
