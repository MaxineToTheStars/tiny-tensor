# Import Statements
# ----------------------------------------------------------------
from typing import Callable, TypeVar

# ---

# ---

# ----------------------------------------------------------------

# File Docstring
# --------------------------------
# A representation of a single scalar value and its gradient.
#
# @author @MaxineToTheStars <https://github.com/MaxineToTheStars>
# ----------------------------------------------------------------

# https://peps.python.org/pep-0673/
TYPE_VALUE = TypeVar("TYPE_VALUE", bound="Value")


# Class Definitions
class Value:
    # Enums

    # Interfaces

    # Constants

    # Public Variables

    # Private Variables
    _value: float = None
    _operation: str = None
    _currentGradient: float = None
    _parents: set[TYPE_VALUE] = None
    _backwardOperation: Callable = None

    # Constructor
    def __init__(self, value: float, parents: tuple[TYPE_VALUE, TYPE_VALUE] = None, operation: str = None) -> TYPE_VALUE:
        """
        Instances a new Value object.

        @param value - The value to pass
        @param parents - The Value's previous parents
        @param operation - The Value's parents operation
        @return Value - A new Value
        """

        # Store value
        self._value = value

        # Store parents
        self._parents = set(parents) if parents != None else set()

        # Store operation
        self._operation = operation

        # Set gradient
        self._currentGradient = 0.0

        # Weird fix for lambda operations
        self._backwardOperation = lambda: None

    # Public Static Methods

    # Public Inherited Methods
    def zeroGradient(self) -> None:
        """
        Zeros out the current gradient.

        @return None
        """

        # Zero out the gradient
        self._currentGradient = 0.0

    def reluFunction(self) -> TYPE_VALUE:
        # Create the output value
        outputValue: Value = Value(0 if self._value < 0 else self._value, (self,), "ReLU")

        # Create the backward operation
        def _backward() -> None:
            self._currentGradient += (outputValue.getValue() > 0) * outputValue.getCurrentGradient()

        # Set the backward operation
        outputValue._backwardOperation = _backward

        # Return
        return outputValue

    def backwardPropagate(self) -> None:
        """
        Backward propagates the current Value.

        @return None
        """

        # Declare orderedOutput
        orderedOutput: list[Value] = []

        # Declare visitedNodes
        visitedNodes: set[Value] = set()

        # Sort by topological order
        def _buildTopological(value: Value):
            if value not in visitedNodes:
                # Add to visited nodes
                visitedNodes.add(value)

                # Iterate through the parents
                for parentNode in value.getParents():
                    _buildTopological(parentNode)
                orderedOutput.append(value)

        # Sort
        _buildTopological(self)

        # Start off with gradient of 1
        self._currentGradient = 1

        for child in reversed(orderedOutput):
            child._backwardOperation()

    def getParents(self) -> set[TYPE_VALUE]:
        """
        Returns the parents of the Value.

        @return Parents - The Value's parents
        """

        # Return the parents
        return self._parents

    def getValue(self) -> float:
        """
        Returns the Value's value as a Python object.

        @return float - The Value's value as a Python object
        """

        # Return the value
        return self._value

    def getCurrentGradient(self) -> float:
        """
        Returns the Value's current gradient as a Python object.

        @return float - The Value's current gradient as a Python object
        """

        # Return the gradient
        return self._currentGradient

    def getOperation(self) -> str:
        """
        Returns the Value's operation as a Python object.

        @return str - The Value's operation as a Python object
        """

        # Return the operation
        return self._operation

    # Private Static Methods

    # Private Inherited Methods
    def __pow__(self, other: int | float) -> TYPE_VALUE:
        # Only accept int/floats
        assert isinstance(other, (int, float), "Only accepting int/float powers")

        # Create the output value
        outputValue: Value = Value(self._value**other, (self, None), "**{exp}".format(exp=other))

        # Create the backward operation
        def _backward() -> None:
            self._currentGradient += (other * self.getValue() ** (other - 1)) * outputValue.getCurrentGradient()

        # Set the backward operation
        outputValue._backwardOperation = _backward

        # Return
        return outputValue

    def __mul__(self, other: object) -> TYPE_VALUE:
        # Wrap in Value if not Value
        wrappedValue: Value = other if isinstance(other, Value) else Value(other)

        # Create the output value
        outputValue: Value = Value((self._value + wrappedValue._value), (self, wrappedValue), "*")

        # Create the backward operation
        def _backward() -> None:
            self._currentGradient += wrappedValue.getValue() * outputValue.getCurrentGradient()
            wrappedValue._currentGradient += self.getValue() * outputValue.getCurrentGradient()

        # Set the backward operation
        outputValue._backwardOperation = _backward

        # Return
        return outputValue

    def __rmul__(self, other: object) -> TYPE_VALUE:
        return self * other

    def __truediv__(self, other: object) -> TYPE_VALUE:
        return self * other**-1

    def __rtruediv__(self, other: object) -> TYPE_VALUE:
        return other * self**-1

    def __add__(self, other: object) -> TYPE_VALUE:
        # Wrap in Value if not Value
        wrappedValue: Value = other if isinstance(other, Value) else Value(other)

        # Create the output value
        outputValue: Value = Value((self._value + wrappedValue._value), (self, wrappedValue), "+")

        # Create the backward operation
        def _backward() -> None:
            self._currentGradient += outputValue.getCurrentGradient()
            wrappedValue._currentGradient += outputValue.getCurrentGradient()

        # Set the backward operation
        outputValue._backwardOperation = _backward

        # Return
        return outputValue

    def __radd__(self, other: object) -> TYPE_VALUE:
        return self + other

    def __sub__(self, other: object) -> TYPE_VALUE:
        return self + (-other)

    def __rsub__(self, other: object) -> TYPE_VALUE:
        return other + (-self)

    def __repr__(self) -> str:
        return f"Value(value={self.getValue()}, gradient={self.getCurrentGradient()})"
