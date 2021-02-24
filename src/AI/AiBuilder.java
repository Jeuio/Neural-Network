package AI;

import AI.Components.Node;
import AI.Math.MathFunctions;

import java.util.ArrayList;

public class AiBuilder {

    private SquishificationFunction squishificationFunction; //Specifies the function used to refine the values of the Nodes
    private ArrayList<Node[]> nodes = new ArrayList<>(); //List for all node arrays
    private Node[] inputLayer; //Array of all nodes in the input layer
    private Node[] outputLayer; //Array of all nodes in the output layer
    private float learningRate; //Learning rate of the network
    private boolean useBias;

    /**
     * Constructor to create a neural network
     * @param layers     amount of layer the network should have
     * @param parameters the amount of nodes the layers should have. Size of this parameter has to be equal to the size of the layers
     */
    public AiBuilder(int layers, boolean useBias, int... parameters) {
        this.useBias = useBias;
        for (byte i = 0; i < layers; i++) { //Creates nodes for all the layers specified in "layers"
            Node[] n = new Node[parameters[i]];
            for (int j = 0; j < n.length; j++) { //Creates the right amount of nodes for the number specified in "parameters"
                Node newNode = new Node(); //Creates a new object of the Node class
                newNode.setLayerPosition(i); //Sets the layer position to the right value

                if (i > 0) { //Check whether the the node is in the input layer
                    float[] weights = new float[nodes.get(i - 1).length]; //Create an array with the size of the amount of nodes in the previous layer
                    for (int k = 0; k < weights.length; k++) {
                        weights[k] = (float) (Math.random() * 2 - 1); //Generate a random weight
                    }
                    newNode.setWeights(weights); //Add the weight array to the node
                }

                if (this.useBias) {
                    newNode.setBias((float) (Math.random() * 2  - 1)); //Assigns a random bias to the node
                }
                n[j] = newNode; //Adds the new node to the node array
            }
            this.nodes.add(n); //Adds the node array to the layer list
        }
        inputLayer = this.nodes.get(0); //Assigns the first layer to "inputLayer"
        outputLayer = this.nodes.get(this.nodes.size() - 1); //Assigns the last Layer to "outputLayer"
    }

    /**
     * Links all the nodes in the network by calling the link method
     */
    public void build() {
        for (Node node : //Goes through all nodes in the output layer
                outputLayer) {
            link(node);
        }
    }

    /**
     * Recursive method to connect all the nodes together
     * @param n the node all nodes in the previous layer should be linked to
     */
    private void link(Node n) {
        if (n.getLayerPosition() > 0) { //Check whether the node is in the first layer (no input nodes can be specified in that case)
            n.setInputNodes(this.nodes.get(n.getLayerPosition() - 1)); //Assigns all the nodes from the previous layer as input nodes
            for (Node node : //Does the same for all nodes in the previous layer
                    this.nodes.get(n.getLayerPosition() - 1)) {
                link(node);
            }
        }
    }

    /**
     * Assigns values to the input layer nodes
     * @param values an array of values to be assigned
     */
    public void assignValues(float[] values) {
        if (values.length != inputLayer.length) { //Checks whether the length of the values array is equal to the amount of nodes in the input layer and throws an exception otherwise
            throw new IllegalArgumentException();
        }
        for (int i = 0; i < values.length; i++) {
            inputLayer[i].setValue(values[i]); //Assign the value of the value array to the nodes
        }
    }

    /**
     * Propagates all the values in the neural network by calling the propagate method
     */
    public void calculateValues() {
        for (Node node :
                outputLayer) {
            node.setValue(propagate(node));
        }
    }

    /**
     * Recursive method to propagate all the values through the network
     * @param node the node the values are propagated from
     * @return the activation of the node
     */
    private float propagate(Node node) {
        if (node.getLayerPosition() > 0) { //Check whether node is in the input layer
            float value = 0;
            float[] weights = node.getWeights(); //Array of all the weights connected to the node
            for (int i = 0; i < node.getInputNodes().length; i++) { //Iterate through all the input nodes of the node
                if (this.useBias) {
                    value += propagate(node.getInputNodes()[i]) * weights[i] + node.getInputNodes()[i].getBias(); //Calculate the activation of the node
                } else {
                    value += propagate(node.getInputNodes()[i]) * weights[i];
                    //System.out.println(value);
                }
            }
            if (this.squishificationFunction == SquishificationFunction.RELU) {
                value = MathFunctions.relu(value);
            } else if (this.squishificationFunction == SquishificationFunction.LEAKYRELU) {
                value = MathFunctions.leakyRelu(value, (float) -0.01, false);
            }
            return value;
        }
        return node.getValue();
    }

    /**
     * Method that lets the network learn
     * @param data an array of data to learn
     * @param expectedOutput the expected output for each data segment
     */
    public void learn(ArrayList<float[]> data, ArrayList<float[]> expectedOutput) {
        for (int i = 0; i < data.size(); i++) {
            assignValues(data.get(i));
            calculateValues();
            calculateCostForNodes(this.nodes.size() - 1, expectedOutput.get(i));
            printCost();
            backpropagate(0);
        }
    }

    private void printCost() {
        float sumOfCost = 0;
        for (Node node:
             outputLayer) {
            sumOfCost += node.getCost();
        }
        System.out.println("Network Cost: " + sumOfCost);
    }

    /**
     * Recursive function that applies the backpropagation algorithm
     * @param layer the layer the weights are adjusted at. Starting value is 0
     * @implNote calculateCostForNodes has to be called to update the cost of the nodes
     */
    private void backpropagate(int layer) {
        for (Node node : //Iterates through all nodes in the layer
                nodes.get(layer)) {
            for (int i = 0; i < nodes.get(layer + 1).length; i++) { //Iterated through all nodes of the subsequent layer
                float[] weights = this.nodes.get(layer + 1)[i].getWeights(); //Weights connected to the node
                System.out.println(learningRate * nodes.get(layer + 1)[i].getCost());
                weights[i] -= learningRate * nodes.get(layer + 1)[i].getCost(); //Adjusts the weight at that point
            }
        }
        if (layer < nodes.size() - 2) { //Checks whether the output layer has been reached
            backpropagate(layer + 1); //Calls the method to calculate the values of the next layer
        }
    }

    /**
     * Method that calculates the cost of the output layer nodes
     * @param layer          the layer of which the cost of the nodes is calculated
     * @param expectedOutput an array of the expected output. Size has to match the number of output nodes
     */
    private void calculateCostForNodes(int layer, float[] expectedOutput) {
        for (int i = 0; i < this.nodes.get(layer).length; i++) { //Iterates through all nodes of the layer
            Node node = this.nodes.get(layer)[i];
            node.setCost((float) (Math.pow(node.getValue() - expectedOutput[i], 2) * MathFunctions.leakyRelu(node.getValue(), (float) -0.01, true))); //Sets the cost of the node
        }
        if (layer > 0) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer); //Calls the other method to calculate the cost for the other layers
        }
    }

    /**
     * Recursive methos that calculate the cost of nodes in a specified layer
     * @param layer the layer of which the cost of the nodes is calculated
     */
    private void calculateCostForNodes(int layer) {
        for (int i = 0; i <this.nodes.get(layer - 1).length; i++) {
            Node node = this.nodes.get(layer - 1)[i];
            float sum = 0; //Sum of the cost
            for (int j = 0; j < nodes.get(layer).length; j++) { //Iterated through all nodes of the layer
                float[] weights = this.nodes.get(layer)[j].getWeights(); //Weights of the node
                sum += nodes.get(layer)[j].getValue() * weights[j]; //Adds the cost of the node to the sum
            }
            node.setCost(sum * MathFunctions.leakyRelu(node.getValue(), (float) -0.01, true)); //Sets the cost of the node
        }
        if (layer > 1) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer - 1); //Call the method to calculate the cost of the previous layer
        }
    }

    /**
     * Methos that return the values of the output layer activation
     * @return the values of the output layer activation
     */
    public float[] getOutputLayerValues() {
        float[] values = new float[this.outputLayer.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = this.outputLayer[i].getValue();
        }
        return values;
    }

    /**
     * Method that return the "squishifiaction" function of the network
     * @return the "squishification" function the network is using
     */
    public SquishificationFunction getSquishificationFunction() {
        return squishificationFunction;
    }

    /**
     * Method that sets the "squishification" function of the network
     * @param squishificationFunction the "squishification" function the neural network should use
     */
    public void setSquishificationFunction(SquishificationFunction squishificationFunction) {
        this.squishificationFunction = squishificationFunction;
    }

    /**
     * Sets the learning rate of the network.
     * @param learningRage the learning rate the network should use. Too high values can result in the network overshooting its gradient in the cost function. Too low values can result in a too small learning effect.
     */
    public void setLearningRate(float learningRage) {
        this.learningRate = learningRage;
    }
}