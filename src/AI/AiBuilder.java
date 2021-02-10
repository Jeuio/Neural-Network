package AI;

import AI.Components.Node;
import AI.Math.MathFunctions;

import java.util.ArrayList;

public class AiBuilder {

    private SquishificationFunction squishificationFunction; //Specifies the function used to refine the values of the Nodes
    private ArrayList<Node[]> nodes = new ArrayList<>(); //List for all node arrays
    private Node[] inputLayer; //Array of all nodes in the input layer
    private Node[] outputLayer; //Array of all nodes in the output layer

    //Constructor which takes the amount of layers and the amount of nodes in those layers as parameters
    public AiBuilder(int layers, int... parameters) {
        for (byte i = 0; i < layers; i++) { //Creates nodes for all the layers specified in "layers"
            Node[] n = new Node[parameters[i]];
            for (int j = 0; j < n.length; j++) { //Creates the right amount of nodes for the number specified in "parameters"
                Node newNode = new Node(); //Creates a new object of the Node class
                newNode.setLayerPosition(i); //Sets the layer position to the right value
                newNode.setWeight((float)(Math.random() * 2 - 1)); //Assigns a random weight to the node
                newNode.setBias((float)(Math.random() * 2 - 1)); //Assigns a random bias to the node
                n[j] = newNode; //Adds the new node to the node array
            }
            this.nodes.add(n); //Adds the node array to the layer list
        }
        inputLayer = this.nodes.get(0); //Assigns the first layer to "inputLayer"
        outputLayer = this.nodes.get(this.nodes.size() - 1); //Assigns the last Layer to "outputLayer"
    }

    //Links all the nodes in the output layer by calling the link method
    public void build() {
        for (Node node: //Goes through all nodes in the output layer
             outputLayer) {
            link(node);
        }
    }

    //Recursive method to connect all nodes together
    private void link(Node n) {
        if (n.getLayerPosition() > 0) { //Check whether the node is in the first layer (no input nodes can be specified in that case)
            n.setInputNodes(this.nodes.get(n.getLayerPosition() - 1)); //Assigns all the nodes from the previous layer as input nodes
            for (Node node: //Does the same for all nodes in the previous layer
                    this.nodes.get(n.getLayerPosition() - 1)) {
                link(node);
            }
        }
    }

    //Assigns values to the input layer
    public void assignValues(float[] values) {
        if (values.length != inputLayer.length) { //Checks whether the length of the values array is equal to the amount of nodes in the input layer and throws an exception otherwise
            throw new IllegalArgumentException();
        }
        for (int i = 0; i < values.length; i++) {
            inputLayer[i].setValue(values[i]); //Assign the value of the value array to the nodes
        }
    }


    //Propagates all the values in the network by calling the propagate method
    public void calculateValues() {
        for (Node node:
             outputLayer) {
            propagate(node);
        }
    }

    //Recursive method to propagate all the values to the output layer
    private float propagate(Node node) {
        if (node.getLayerPosition() > 0) { //Check whether node is in the input layer
            float value = 0;
            for (Node inputNode: //Go through all input nodes of that node and calculate the total value
                 node.getInputNodes()) {
                value += propagate(inputNode) * inputNode.getWeight() + inputNode.getBias();
            }
            if (this.squishificationFunction == SquishificationFunction.RELU) {
                value = MathFunctions.relu(value);
            } else if (this.squishificationFunction == SquishificationFunction.LEAKYRELU) {
                value = MathFunctions.leakyRelu(value, (float)-0.01);
            }
            return value;
        }
        return node.getValue() * node.getWeight() + node.getBias();
    }


    //Method to return the values of the output layer
    public float[] getOutputLayerValues() {
        float[] values = new float[this.outputLayer.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = this.outputLayer[i].getValue();
        }
        return values;
    }

    //Getter and setter for the function to refine the values

    public SquishificationFunction getSquishificationFunction() {
        return squishificationFunction;
    }

    public void setSquishificationFunction(SquishificationFunction squishificationFunction) {
        this.squishificationFunction = squishificationFunction;
    }
}