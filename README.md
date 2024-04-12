<!-- Move text down -->
<br>

<!-- Header -->
<h1 align="center"><sup><sup>Tiny</sup></sup> Tensor</h1>

<!-- Description -->
<h3 align="center">An example AI/ML framework to show the core fundamentals of AI/ML development</h3>
<h4 align="center">Note: No tensors are involved in this project 😝</h4>

## Example
```py
# Imports
from src.classes.value import Value
from src.classes.nn import Neuron

# Create a neuron with 2 inputs
neuron = Neuron(2)

# Create some dummy data
x = [Value(1), Value(-2)]

# Activate the neuron
y = neuron(x)

# Backward Propagate
y.backwardPropagate()
```

## Setup
The following example is written for a Linux system but it is similar for Windows/MacOS.
```bash
# Clone the repository
git clone https://github.com/MaxineToTheStars/tiny-tensor.git

# Switch to the new project directory
cd ./tiny-tensor

# Create a virtual environment and activate it
python3 -m venv .env && source ./.env/bin/activate

# Download the requisite packages
pip3 install -r ./lib/requirements.txt

# Switch directories and run the example
cd ./lib && python3 example.py
```

## Project Layout
```
lib/src/classes/nn.py       <---> A basic NN framework
lib/src/classes/value.py    <---> A scalar value engine
lib/src/example.py          <---> An example project
```
