# Import Statements
# ----------------------------------------------------------------
import random
from typing import TypeVar

# ---
from .value import Value

# ---

# ----------------------------------------------------------------

# File Docstring
# --------------------------------
# A basic NN framework.
#
# @author @MaxineToTheStars <https://github.com/MaxineToTheStars>
# ----------------------------------------------------------------

# https://peps.python.org/pep-0673/
TYPE_BASE = TypeVar("TYPE_BASE", bound="Base")
TYPE_NEURON = TypeVar("TYPE_NEURON", bound="Neuron")
TYPE_LAYER = TypeVar("TYPE_LAYER", bound="Layer")
TYPE_MLP = TypeVar("TYPE_MLP", bound="MLP")


# Class Definitions
class Base:
    # Enums

    # Interfaces

    # Constants

    # Public Variables

    # Private Variables

    # Constructor

    # Public Static Methods

    # Public Inherited Methods
    def zeroGradients(self) -> None:
        """
        Zeros out the gradients in all parameters.

        @return None
        """

        # Get all parameters
        for param in self.getParameters():
            # Zero the gradient
            param.zeroGradient()

    def getParameters(self) -> list[Value]:
        """
        Return the parameters for this NeuralNet.

        @return Parameters - An array of Values
        """

        # Return
        return []

    # Private Static Methods

    # Private Inherited Methods


class Neuron(Base):
    # Enums

    # Interfaces

    # Constants

    # Public Variables

    # Private Variables
    _neuronBias: Value = None
    _neuronWeights: list[Value] = None

    # Constructor
    def __init__(self, numberOfInputs: int) -> TYPE_NEURON:
        """
        Instances a new Neuron object.

        @param numberOfInputs - How many inputs this neuron can receive
        @return Neuron - A new Neuron object
        """

        # Generate an array of neuron weights
        self._neuronWeights = [Value(random.uniform(-1, 1)) for _ in range(0, numberOfInputs)]

        # Set a bias
        self._neuronBias = Value(0)

    # Public Static Methods

    # Public Inherited Methods
    def getParameters(self) -> list[Value]:
        """
        Return the parameters for this NeuralNet.

        @return Parameters - An array of Values
        """

        # Return the weights added to the bias
        return self._neuronWeights + [self._neuronBias]

    # Private Static Methods

    # Private Inherited Methods
    def __call__(self, x: list[float]) -> Value:
        """
        Dark voodo magic

        @param x - ???
        @return None
        """

        # ???
        return sum((wi * xi for wi, xi in zip(self._neuronWeights, x)), self._neuronBias)


class Layer(Base):
    # Enums

    # Interfaces

    # Constants

    # Public Variables

    # Private Variables
    _layerNeurons: list[Neuron] = None

    # Constructor
    def __init__(self, numberOfInputNeurons: int, numberOfOutputNeurons: int) -> TYPE_LAYER:
        """
        Instances a new Layer object.

        @param numberOfInputNeurons - The number of input Neurons
        @param numberOfOutputNeurons - The number of output Neurons
        @return Layer - A Layer of Neurons
        """

        # Create a layer of neurons
        self._layerNeurons = [Neuron(numberOfInputNeurons) for _ in range(0, numberOfOutputNeurons)]

    # Public Static Methods

    # Public Inherited Methods
    def getParameters(self) -> list[Value]:
        """
        Return the parameters for this NeuralNet.

        @return Parameters - An array of Values
        """

        # Get and return parameters
        return [parameter for neuron in self._layerNeurons for parameter in neuron.getParameters()]

    # Private Static Methods

    # Private Inherited Methods
    def __call__(self, x: float) -> list[Value] | Value:
        # Get a list of neuron values
        outData: list[Value] = [neuron(x) for neuron in self._layerNeurons]

        # Return either the Value as Value or Value as array
        # Aka: [Value] or Value
        return outData[0] if len(outData) == 1 else outData


class MLP(Base):
    # Enums

    # Interfaces

    # Constants

    # Public Variables

    # Private Variables
    _mlpLayer: list[Layer] = None

    # Constructor
    def __init__(self, numberOfInputs: int, numberOfOutputs: list[int]) -> TYPE_MLP:
        """
        Instances a new MultiLayerPerceptron object.

        @param numberOfInputs - The number of inputs
        @param numberOfOutputs - The number of outputs as an array
        @return MLP - A new MultiLayerPerceptron object
        """

        # Calculate the layer size
        layerSize: list[int] = [numberOfInputs] + numberOfOutputs

        # Create layers
        self._mlpLayer = [Layer(layerSize[i], layerSize[i + 1]) for i in range(0, len(numberOfOutputs))]

    # Public Static Methods

    # Public Inherited Methods
    def getParameters(self) -> list[Value]:
        """
        Return the parameters for this NeuralNet.

        @return Parameters - An array of Values
        """

        # Get and return parameters
        return [parameter for layer in self._mlpLayer for parameter in layer.getParameters()]

    # Private Static Methods

    # Private Inherited Methods
    def __call__(self, x: float):
        for layer in self._mlpLayer:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self._mlpLayer)}]"
