# Import Statements
# ----------------------------------------------------------------
import random

# ---
from src.classes.value import Value
from src.classes.nn import Neuron, Layer, MLP

# ---
from graphviz import Digraph

# ----------------------------------------------------------------

# File Docstring
# --------------------------------
# Some examples.
#
# @author @MaxineToTheStars <https://github.com/MaxineToTheStars>
# ----------------------------------------------------------------


# Class Definitions
class Example1:
    # Enums

    # Interfaces

    # Constants

    # Public Variables

    # Private Variables

    # Constructor

    # Public Static Methods

    # Public Inherited Methods
    def run(self) -> None:
        # Set the seed
        random.seed(1337)

        # Create a neuron with 2 inputs
        neuron = Neuron(2)

        # Create some dummy data
        x = [Value(1), Value(-2)]

        # Activate the neuron
        y = neuron(x)

        # Backward Propagate
        y.backwardPropagate()

        # Render
        self._drawGraph(y).render("out")

    # Private Static Methods

    # Private Inherited Methods
    def _drawGraph(self, root: Value, format: str = "svg", direction: str = "LR") -> Digraph:
        # Check that the direction is valid
        assert direction in ["LR", "TB"]

        # Declare and set nodes and edges
        nodes, edges = self._traceChildrenInGraph(root)

        # Create the diagram
        diagram: Digraph = Digraph(format=format, graph_attr={"rankdir": direction})

        # Populate
        for node in nodes:
            # Create a new node
            diagram.node(name=str(id(node)), label="{ value %.4f | gradient %.4f}" % (node.getValue(), node.getCurrentGradient()), shape="record")

            # Check if it has an operation
            if node.getOperation() != None:
                diagram.node(name=(str((id(node))) + node.getOperation()), label=node.getOperation())
                diagram.edge((str(id(node)) + node.getOperation()), str(id(node)))

        for node1, node2 in edges:
            diagram.edge(str(id(node1)), (str(id(node2)) + node2.getOperation()))

        return diagram

    def _traceChildrenInGraph(self, rootNode: Value) -> tuple[set[Value], set]:
        # Declare nodes and edges
        nodes, edges = set(), set()

        # Declare build method
        def _buildGraphData(startValue: Value) -> None:
            if startValue not in nodes:
                # Add value to node
                nodes.add(startValue)

                # Iterate through its' parents
                for parentNode in startValue.getParents():
                    # Add a new edge
                    edges.add((parentNode, startValue))

                    # Recursively continue
                    _buildGraphData(parentNode)

        # Build
        _buildGraphData(rootNode)

        # Return nodes and edges
        return nodes, edges


# Pick your poison
Example1().run()
