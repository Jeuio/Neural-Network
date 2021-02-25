package AI;

import AI.Components.Node;
import AI.Math.MathFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Scanner;

public class AiBuilder {

    private SquishificationFunction squishificationFunction; //Specifies the function used to refine the values of the Nodes
    private ArrayList<Node[]> nodes = new ArrayList<>(); //List for all node arrays
    private Node[] inputLayer; //Array of all nodes in the input layer
    private Node[] outputLayer; //Array of all nodes in the output layer
    private double learningRate; //Learning rate of the network
    private boolean useBias;

    /**
     * Constructor to create a neural network
     *
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
                    double[] weights = new double[nodes.get(i - 1).length]; //Create an array with the size of the amount of nodes in the previous layer
                    for (int k = 0; k < weights.length; k++) {
                        weights[k] = randomValue(-Math.sqrt(this.nodes.get(this.nodes.size() - 1).length), Math.sqrt(this.nodes.get(this.nodes.size() - 1).length)); //Generate a random weight
                    }
                    newNode.setWeights(weights); //Add the weight array to the node
                }

                if (this.useBias && i > 0) {
                    newNode.setBias(randomValue(-Math.sqrt(this.nodes.get(this.nodes.size() - 1).length), Math.sqrt(this.nodes.get(this.nodes.size() - 1).length))); //Assigns a random bias to the node
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
     *
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
     *
     * @param values an array of values to be assigned
     */
    public void assignValues(double[] values) {
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
     *
     * @param node the node the values are propagated from
     * @return the activation of the node
     */
    private double propagate(Node node) {
        if (node.getLayerPosition() > 0) { //Check whether node is in the input layer
            double value = 0;
            double[] weights = node.getWeights(); //Array of all the weights connected to the node
            for (int i = 0; i < this.nodes.get(node.getLayerPosition() - 1).length; i++) { //Iterate through all the input nodes of the node
                value += propagate(this.nodes.get(node.getLayerPosition() - 1)[i]) * weights[i]; //Calculate the activation of the node
            }
            if (this.useBias) {
                value += node.getBias();
            }
            if (this.squishificationFunction == SquishificationFunction.RELU) {
                value = MathFunctions.relu(value);
            } else if (this.squishificationFunction == SquishificationFunction.LEAKYRELU) {
                value = MathFunctions.leakyRelu(value, -0.01, false);
            } else if (this.squishificationFunction == SquishificationFunction.SIGMOID) {
                value = MathFunctions.sigmoid(value, false);
            }
            return value;
        } else {
            return node.getValue();
        }
    }

    /**
     * Method that lets the network learn
     *
     * @param data           an array of data to learn
     * @param expectedOutput the expected output for each data segment
     */
    public void learn(int epochs, String filePath, ArrayList<double[]> data, ArrayList<double[]> expectedOutput) {
        File saveFile = new File(filePath);
        if (saveFile.exists()) {
            readProgress(filePath);
        } else {
            try {
                if (saveFile.createNewFile()) {
                    System.out.println("New save file created at " + filePath);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        for (int i = 0; i < epochs; i++) {
            long startTime = System.currentTimeMillis();
            int correctPredictions = 0;
            BigDecimal sumOfCost = new BigDecimal(0);
            for (int j = 0; j < data.size(); j++) {
                assignValues(data.get(j));
                calculateValues();
                calculateCostForNodes(this.nodes.size() - 1, expectedOutput.get(j));
                sumOfCost = sumOfCost.add(BigDecimal.valueOf(networkCost()));
                int biggestValueIndex = 0;
                for (int k = 0; k < expectedOutput.get(j).length; k++) {
                    if (expectedOutput.get(j)[k] > expectedOutput.get(j)[biggestValueIndex]) {
                        biggestValueIndex = k;
                    }
                }
                if (getDecision() == biggestValueIndex) {
                    correctPredictions++;
                }
                backpropagate(0);
            }
            safeProgress(filePath);
            System.out.println("Epoch: " + i + "; Average cost of epoch: " + sumOfCost.divide(BigDecimal.valueOf(i + 1), RoundingMode.CEILING) + "; Percentage correct: " + (double)correctPredictions / (double)data.size() * 100 + "%; Calculation time: " + (System.currentTimeMillis() - startTime) / 1000 + "s");
        }
    }

    private double networkCost() {
        double sumOfCost = 0;
        for (Node node :
                this.outputLayer) {
            sumOfCost += node.getCost();
        }
        return sumOfCost;
    }

    /**
     * Method that saves the values of the weights to a file
     *
     * @param path the path to the file the values should be saved to. Should be a txt file
     */
    private void safeProgress(String path) {
        try {
            FileWriter writer = new FileWriter(new File(path));
            for (int layerIndex = 1; layerIndex < this.nodes.size(); layerIndex++) {
                for (int i = 0; i < this.nodes.get(layerIndex).length; i++) {
                    double[] weights = this.nodes.get(layerIndex)[i].getWeights();
                    for (int j = 0; j < weights.length; j++) {
                        if (j != weights.length - 1) {
                            writer.write(weights[j] + ","); //Symbol for separating individual weight values
                        } else {
                            writer.write(String.valueOf(weights[j]));
                        }
                    }
                    if (i != this.nodes.get(layerIndex).length - 1) {
                        writer.write(";"); //Symbol for separating weights for nodes
                    } else {
                        writer.write("_"); //Symbol for separating weights for layers
                    }
                }
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Method that reads the values of the weights
     *
     * @param path the path to the file the values are written to. Should be a txt file
     */
    private void readProgress(String path) {
        try {
            System.out.println("Loading progress from file");
            Scanner scanner = new Scanner(new File(path));
            if (scanner.hasNextLine()) {
                String weightString = scanner.nextLine();
                String[] layerWeights = weightString.split("_");
                for (int i = 1; i < nodes.size(); i++) {
                    String[] nodeWeights = layerWeights[i - 1].split(";");
                    for (int j = 0; j < nodeWeights.length; j++) {
                        String[] weights = nodeWeights[j].split(",");
                        double[] doubleWeights = new double[weights.length];
                        for (int k = 0; k < weights.length; k++) {
                            doubleWeights[k] = Double.parseDouble(weights[k]);
                        }
                        nodes.get(i)[j].setWeights(doubleWeights);
                    }
                }
                System.out.println("Progress loaded");
            } else {
                System.out.println("Progress file is empty");
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * Recursive function that applies the backpropagation algorithm
     *
     * @param layer the layer the weights are adjusted at. Starting value is 0
     * @implNote calculateCostForNodes has to be called to update the cost of the nodes
     */
    private void backpropagate(int layer) {
        for (int i = 0; i < nodes.get(layer).length; i++) {
            Node node = nodes.get(layer)[i];
            for (int j = 0; j < nodes.get(layer + 1).length; j++) { //Iterated through all nodes of the subsequent layer
                double[] weights = this.nodes.get(layer + 1)[j].getWeights(); //Weights connected to the node
                weights[i] -= this.learningRate * node.getValue() * this.nodes.get(layer + 1)[j].getCost(); //Adjusts the weight at that point
                this.nodes.get(layer + 1)[j].setWeights(weights);
            }
            if (this.useBias) {
                node.setBias(node.getBias() - this.learningRate * node.getCost());
            }
        }
        if (layer < nodes.size() - 2) { //Checks whether the output layer has been reached
            backpropagate(layer + 1); //Calls the method to calculate the values of the next layer
        }
    }

    /**
     * Method that calculates the cost of the output layer nodes
     *
     * @param layer          the layer of which the cost of the nodes is calculated
     * @param expectedOutput an array of the expected output. Size has to match the number of output nodes
     */
    private void calculateCostForNodes(int layer, double[] expectedOutput) {
        for (int i = 0; i < this.nodes.get(layer).length; i++) { //Iterates through all nodes of the layer
            Node node = this.nodes.get(layer)[i];
            node.setCost((node.getValue() - expectedOutput[i]) * node.getValue() * (1 - node.getValue())); //Sets the cost of the node
        }
        if (layer > 0) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer); //Calls the other method to calculate the cost for the other layers
        }
    }

    /**
     * Recursive methos that calculate the cost of nodes in a specified layer
     *
     * @param layer the layer of which the cost of the nodes is calculated
     */
    private void calculateCostForNodes(int layer) {
        for (int i = 0; i < this.nodes.get(layer - 1).length; i++) {
            Node node = this.nodes.get(layer - 1)[i];
            double sum = 0; //Sum of the cost
            for (int j = 0; j < nodes.get(layer).length; j++) { //Iterated through all nodes of the layer
                double[] weights = this.nodes.get(layer)[j].getWeights(); //Weights of the node
                sum += nodes.get(layer)[j].getCost() * weights[j]; //Adds the cost of the node to the sum
            }
            node.setCost(sum * node.getValue() * (1 - node.getValue())); //Sets the cost of the node
        }
        if (layer > 2) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer - 1); //Call the method to calculate the cost of the previous layer
        }
    }

    /**
     * Methos that return the values of the output layer activation
     *
     * @return the values of the output layer activation
     */
    public double[] getOutputLayerValues() {
        double[] values = new double[this.outputLayer.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = this.outputLayer[i].getValue();
        }
        return values;
    }

    public int getDecision() {
        double[] values = getOutputLayerValues();
        double biggestValue = -1000;
        int biggestValueIndex = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > biggestValue) {
                biggestValue = values[i];
                biggestValueIndex = i;
            }
        }
        return biggestValueIndex;
    }

    private double randomValue(double lowerLimit, double upperLimit) {
        return Math.random() * (upperLimit - lowerLimit + 1) + lowerLimit;
    }

    /**
     * Method that return the "squishifiaction" function of the network
     *
     * @return the "squishification" function the network is using
     */
    public SquishificationFunction getSquishificationFunction() {
        return squishificationFunction;
    }

    /**
     * Method that sets the "squishification" function of the network
     *
     * @param squishificationFunction the "squishification" function the neural network should use
     */
    public void setSquishificationFunction(SquishificationFunction squishificationFunction) {
        this.squishificationFunction = squishificationFunction;
    }

    /**
     * Sets the learning rate of the network.
     *
     * @param learningRage the learning rate the network should use. Too high values can result in the network adjusting the weight to violently and overshooting its target. Too low values can result in a too small learning effect.
     */
    public void setLearningRate(double learningRage) {
        this.learningRate = learningRage;
    }
}