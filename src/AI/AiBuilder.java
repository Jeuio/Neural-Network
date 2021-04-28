package AI;

import AI.Components.Bias;
import AI.Components.Layer;
import AI.Components.Node;
import AI.Components.Weight;
import AI.Cost.CostFunction;
import AI.Cost.CostFunctions;
import AI.Learning.Minibatch;
import AI.Learning.MinibatchCreator;
import AI.Math.MathFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * This class is used to build and manage neural networks
 *
 * @// TODO: 28.03.2021 make a class that handles the learning and value-extraction aspect of this class, since it is getting to big
 */
public class AiBuilder {

    private Layer inputLayer; //Array of all nodes in the input layer
    private Layer outputLayer; //Array of all nodes in the output layer
    private double cost = 0; //The cost of the network
    private double learningRate; //Learning rate of the network
    private boolean useBias = true; //Whether the network uses a bias
    private CostFunctions costFunction; //The cost function of the network
    private final ArrayList<Layer> layers = new ArrayList<>(); //Stores the layers of the network


    public int guess(double[] data) {
        this.assignValues(data);
        this.propagate();
        return this.getDecision();
    }

    /**
     * This method is used to build the neural network. It creates all nodes, weights, and biases and assigns all of them accordingly.
     * It also invokes the link method to connect all nodes and weights together
     */
    public void build() {
        for (byte i = 0; i < this.layers.size(); i++) { //Creates nodes for all the layers specified in "layers"
            Node[] nodeArray = new Node[this.layers.get(i).getLayerSize()];
            for (int j = 0; j < nodeArray.length; j++) { //Creates the right amount of nodes for the number specified in "parameters"
                Node newNode = new Node(); //Creates a new object of the Node class
                newNode.setLayerPosition(i); //Sets the layer position to the right value
                newNode.setIndexInLayer(j);

                if (i > 0) { //Check whether the the node is in the input layer
                    Weight[] weights = new Weight[this.layers.get(i - 1).getLayerSize()]; //Create an array with the size of the amount of nodes in the previous layer
                    for (int k = 0; k < weights.length; k++) {
                        weights[k] = new Weight(randomValue(-Math.sqrt(this.layers.get(i - 1).getLayerSize()), Math.sqrt(this.layers.get(i - 1).getLayerSize()))); //Generate a random weight
                    }
                    newNode.setWeights(weights); //Add the weight array to the node
                }

                if (this.useBias && i > 0) {
                    newNode.setBias(new Bias(randomValue(-Math.sqrt(this.layers.get(i - 1).getLayerSize()), Math.sqrt(this.layers.get(i - 1).getLayerSize())))); //Assigns a bias to the node
                }
                nodeArray[j] = newNode; //Adds the new node to the node array
            }
            this.layers.get(i).setNodes(nodeArray); //Adds the node array to the layer list
        }
        inputLayer = this.layers.get(0); //Assigns the first layer to "inputLayer"
        outputLayer = this.layers.get(this.layers.size() - 1); //Assigns the last Layer to "outputLayer"

        link();
    }

    /**
     * This method connects all nodes and weights together
     */
    private void link() {
        for (int layerIndex = 1; layerIndex < this.layers.size(); layerIndex++) {
            Layer leftLayer = this.layers.get(layerIndex - 1);
            Layer rightLayer = this.layers.get(layerIndex);
            for (int rightNodeIndex = 0; rightNodeIndex < rightLayer.getLayerSize(); rightNodeIndex++) {
                Node rightNode = rightLayer.getNodes()[rightNodeIndex];
                rightNode.setInputNodes(leftLayer.getNodes());
            }
        }
    }

    /**
     * Assigns values to the input layer nodes
     *
     * @param values an array of values to be assigned
     */
    public void assignValues(double[] values) {
        if (values.length != this.layers.get(0).getLayerSize()) { //Checks whether the length of the values array is equal to the amount of nodes in the input layer and throws an exception otherwise
            throw new IllegalArgumentException();
        }
        for (int i = 0; i < values.length; i++) {
            inputLayer.getNodes()[i].setValue(values[i]); //Assign the value of the value array to the nodes
            inputLayer.getNodes()[i].setActivation(values[i]);
        }
    }

    /**
     * Method to propagate all the values through the network
     */
    private void propagate() {
        for (int layerIndex = 1; layerIndex < this.layers.size(); layerIndex++) {
            Layer outputLayer = this.layers.get(layerIndex);
            for (int outputIndex = 0; outputIndex < outputLayer.getNodes().length; outputIndex++) {
                Node outputNode = outputLayer.getNodes()[outputIndex];
                double value = 0;
                Weight[] weights = outputNode.getWeights();
                for (int inputIndex = 0; inputIndex < outputNode.getInputNodes().length; inputIndex++) {
                    value += outputNode.getInputNodes()[inputIndex].getActivation() * weights[inputIndex].getValue();
                }
                if (this.useBias) {
                    value += outputNode.getBias().getValue();
                }

                outputNode.setValue(value);

                if (outputLayer.getActivationFunction() == ActivationFunction.RELU) {
                    value = MathFunctions.relu(value, false);
                } else if (outputLayer.getActivationFunction() == ActivationFunction.LEAKYRELU) {
                    value = MathFunctions.leakyRelu(value, -0.01, false);
                } else if (outputLayer.getActivationFunction() == ActivationFunction.SIGMOID) {
                    value = MathFunctions.sigmoid(value, false);
                } else if (outputLayer.getActivationFunction() == ActivationFunction.SOFTMAX) {
                    value = MathFunctions.softmax(outputNode.getIndexInLayer(), this.layers.get(outputNode.getLayerPosition()).getNodes(), false);
                }

                outputNode.setActivation(value);
            }
        }
    }

    /**
     * Method that lets the network learn
     *
     * @param epochs         the number of epochs the network should learn before stopping
     * @param minibatchsize  the preferred size of mini batches
     * @param filePath       path to the file where the progress should be stored
     * @param data           an array of data to learn
     * @param expectedOutput the expected output for each data segment
     * @// TODO: 28.03.2021 give the user more freedom to decide whether to use mini batches or whether to safe the progress. The user should have the ability to disable checkSaving
     */
    public void learn(int epochs, int minibatchsize, String filePath, ArrayList<double[]> data, ArrayList<double[]> expectedOutput) {

        MinibatchCreator minibatchCreator = new MinibatchCreator();
        minibatchCreator.setData(data);
        minibatchCreator.setLabels(expectedOutput);

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

            minibatchCreator.createMiniBatches(minibatchsize);
            Minibatch[] minibatches = minibatchCreator.getMinibatches();
            for (int j = 0; j < minibatches.length; j++) {
                Minibatch minibatch = minibatches[j];
                for (int k = 0; k < minibatch.getData().size(); k++) {
                    assignValues(minibatch.getData().get(k));
                    propagate();
                    calculateCostForNodes(outputLayer.getLayerPosition(), minibatch.getLabels().get(k));

                    sumOfCost = sumOfCost.add(BigDecimal.valueOf(this.cost));
                    this.cost = 0;

                    int biggestValueIndex = 0;
                    for (int l = 0; l < minibatch.getLabels().get(k).length; l++) {
                        if (minibatch.getLabels().get(k)[l] > minibatch.getLabels().get(k)[biggestValueIndex]) {
                            biggestValueIndex = l;
                        }
                    }
                    if (getDecision() == biggestValueIndex) {
                        correctPredictions++;
                    }
                    backpropagate();
                }
                applyChanges();
            }
            checkSaving(filePath);
            System.out.println("Epoch: " + i + "; Average cost of epoch: " + sumOfCost.divide(BigDecimal.valueOf(minibatches.length), RoundingMode.FLOOR) + "; Percentage correct: " + (double) correctPredictions / (double) data.size() * 100 + "%; Calculation time: " + (System.currentTimeMillis() - startTime) / 1000 + "s");
        }
    }

    /**
     * @return the cost of the network
     */
    private double networkCost() {
        return this.cost;
    }

    /**
     * Checks whether saving was successful
     *
     * @param filePath the path of the safe file
     */
    public void checkSaving(String filePath) {
        ArrayList<ArrayList<Weight[]>> layerWeights1 = new ArrayList<>();
        for (int i = 1; i < this.layers.size(); i++) {
            ArrayList<Weight[]> weights = new ArrayList<>();
            Layer layer = this.layers.get(i);
            for (Node node :
                    layer.getNodes()) {
                Weight[] w = node.getWeights();
                weights.add(w);
            }
            layerWeights1.add(weights);
        }
        safeProgress(filePath);
        readProgress(filePath);

        ArrayList<ArrayList<Weight[]>> layerWeights2 = new ArrayList<>();
        for (int i = 1; i < this.layers.size(); i++) {
            ArrayList<Weight[]> weights = new ArrayList<>();
            Layer layer = this.layers.get(i);
            for (Node node :
                    layer.getNodes()) {
                Weight[] w = node.getWeights();
                weights.add(w);
            }
            layerWeights2.add(weights);
        }

        for (int i = 0; i < layerWeights1.size(); i++) {
            for (int j = 0; j < layerWeights1.get(i).size(); j++) {
                for (int k = 0; k < layerWeights1.get(i).get(j).length; k++) {
                    if (layerWeights1.get(i).get(j)[k].getValue() != layerWeights2.get(i).get(j)[k].getValue()) {
                        System.out.println("no match at " + i + " " + j + " " + k);
                    }
                }
            }
        }
    }

    /**
     * Method that saves the values of the weights and biases to a file
     *
     * @param path the path to the file the values should be saved to. Should be a txt file
     */
    private void safeProgress(String path) {
        try {
            FileWriter writer = new FileWriter(path);
            for (int layerIndex = 1; layerIndex < this.layers.size(); layerIndex++) {
                for (int i = 0; i < this.layers.get(layerIndex).getLayerSize(); i++) {
                    Weight[] weights = this.layers.get(layerIndex).getNodes()[i].getWeights();
                    for (int j = 0; j < weights.length; j++) {
                        if (j != weights.length - 1) {
                            writer.write(weights[j].getValue() + ","); //Symbol for separating individual weight values
                        } else {
                            writer.write(String.valueOf(weights[j].getValue()));
                        }
                    }
                    if (i != this.layers.get(layerIndex).getLayerSize() - 1) {
                        writer.write(";"); //Symbol for separating weights for nodes
                    } else {
                        if (layerIndex != this.layers.size() - 1) {
                            writer.write("_"); //Symbol for separating weights for layers
                        }
                    }
                }
            }
            writer.write("\r\n");
            if (this.useBias) {
                for (int layerIndex = 1; layerIndex < this.layers.size(); layerIndex++) {
                    for (int i = 0; i < this.layers.get(layerIndex).getLayerSize(); i++) {
                        Node node = this.layers.get(layerIndex).getNodes()[i];
                        if (i != this.layers.get(layerIndex).getLayerSize() - 1) {
                            writer.write(node.getBias().getValue() + ",");
                        } else {
                            writer.write(String.valueOf(node.getBias().getValue()));
                        }
                    }
                    if (layerIndex != this.layers.size() - 1) {
                        writer.write(";");
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
            System.out.println("Loading progress from file...");
            Scanner scanner = new Scanner(new File(path));
            if (scanner.hasNextLine()) {
                String weightString = scanner.nextLine();
                String[] layerWeights = weightString.split("_");
                for (int i = 1; i < this.layers.size(); i++) {
                    String[] nodeWeights = layerWeights[i - 1].split(";");
                    for (int j = 0; j < nodeWeights.length; j++) {
                        String[] stringWeights = nodeWeights[j].split(",");
                        Weight[] weights = new Weight[stringWeights.length];
                        for (int k = 0; k < weights.length; k++) {
                            weights[k] = new Weight(Double.parseDouble(stringWeights[k]));
                        }
                        this.layers.get(i).getNodes()[j].setWeights(weights);
                    }
                }
                System.out.println("Weights loaded");
            } else {
                System.out.println("Progress file is empty");
                return;
            }
            if (scanner.hasNextLine()) {
                String biasString = scanner.nextLine();
                String[] layerString = biasString.split(";");
                for (int i = 1; i < this.layers.size(); i++) {
                    String[] nodeString = layerString[i - 1].split(",");
                    for (int j = 0; j < this.layers.get(i).getLayerSize(); j++) {
                        Node node = this.layers.get(i).getNodes()[j];
                        node.setBias(new Bias(Double.parseDouble(nodeString[j])));
                    }
                }
                System.out.println("Biases loaded");
            } else {
                System.out.println("No biases specified in file");
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * Method that applies the backpropagation algorithm to determine the negative gradient of the cost function in respect to each weight ans bias in the network.
     * The values first need to be propagated through the network and the cost needs to be determined in order to use this algorithm
     */
    private void backpropagate() {
        for (int layerIndex = 0; layerIndex < this.layers.size() - 1; layerIndex++) {
            for (int rightNodeIndex = 0; rightNodeIndex < this.layers.get(layerIndex + 1).getLayerSize(); rightNodeIndex++) {
                Node rightNode = this.layers.get(layerIndex + 1).getNodes()[rightNodeIndex];
                for (int leftNodeIndex = 0; leftNodeIndex < this.layers.get(layerIndex).getLayerSize(); leftNodeIndex++) {
                    Node leftNode = this.layers.get(layerIndex).getNodes()[leftNodeIndex];

                    double gradientOfWeight = rightNode.getCost() * leftNode.getActivation();

                    rightNode.getWeights()[leftNodeIndex].addChange(gradientOfWeight);
                }
                if (this.useBias) {

                    double gradientOfBias = rightNode.getCost();

                    rightNode.getBias().addChange(gradientOfBias);
                }
            }
        }
    }

    /**
     * Method that calculates the cost of the nodes in the output layer. Invokes the method of calculating the cost of the nodes in the hidden layer afterwards
     *
     * @param layer          the layer of which the cost of the nodes is calculated
     * @param expectedOutput an array of the expected output. It's size has to match the number of output nodes
     */
    private void calculateCostForNodes(int layer, double[] expectedOutput) {
        for (int outputIndex = 0; outputIndex < this.layers.get(layer).getLayerSize(); outputIndex++) { //Iterates through all nodes of the layer
            Node outputNode = this.layers.get(layer).getNodes()[outputIndex];
            double cost;

            if (this.costFunction == CostFunctions.CROSS_ENTROPY) {
                cost = CostFunction.derivativeCrossEntropy(this.layers.get(layer).getNodes(), expectedOutput)[outputNode.getIndexInLayer()];
                this.cost += CostFunction.crossEntropy(this.layers.get(layer).getNodes(), expectedOutput);
            } else {
                cost = expectedOutput[outputIndex] - outputNode.getActivation();
                if (this.layers.get(layer).getActivationFunction() == ActivationFunction.SIGMOID) {
                    cost *= MathFunctions.sigmoid(outputNode.getValue(), true);
                } else if (this.layers.get(layer).getActivationFunction() == ActivationFunction.SOFTMAX) {
                    cost *= MathFunctions.softmax(outputNode.getIndexInLayer(), this.layers.get(outputNode.getLayerPosition()).getNodes(), true);
                }
            }
            outputNode.setCost(cost); //Sets the cost of the node
        }
        if (layer > 1) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer); //Calls the other method to calculate the cost for the other layers
        }
    }

    /**
     * Method to calculate the cost of the nodes in the hidden layer
     *
     * @param layer the layer of which the cost of the nodes is calculated
     */
    private void calculateCostForNodes(int layer) {
        for (int layerIndex = 0; layerIndex < this.layers.size() - 1; layerIndex++) {
            Layer leftLayer = this.layers.get(layerIndex);
            Layer rightLayer = this.layers.get(layerIndex + 1);
            for (int leftNodeIndex = 0; leftNodeIndex < this.layers.get(layerIndex).getLayerSize(); leftNodeIndex++) {
                Node leftNode = leftLayer.getNodes()[leftNodeIndex];
                double costSum = 0;
                for (int rightNodeIndex = 0; rightNodeIndex < rightLayer.getLayerSize(); rightNodeIndex++) {
                    Node rightNode = rightLayer.getNodes()[rightNodeIndex];
                    Weight[] rightNodeWeights = rightNode.getWeights();
                    costSum += rightNode.getCost() * rightNodeWeights[leftNodeIndex].getValue();
                }
                if (this.layers.get(layer - 1).getActivationFunction() == ActivationFunction.SIGMOID) {
                    cost *= MathFunctions.sigmoid(leftNode.getValue(), true);
                } else if (this.layers.get(layer - 1).getActivationFunction() == ActivationFunction.SOFTMAX) {
                    cost *= MathFunctions.softmax(leftNode.getIndexInLayer(), leftLayer.getNodes(), true); //Something might be wrong here
                } else if (this.layers.get(layer - 1).getActivationFunction() == ActivationFunction.RELU) {
                    cost *= MathFunctions.relu(leftNode.getActivation(), true);
                }
                leftNode.setCost(costSum);
            }
        }
    }

    /**
     * Method that return the values of the output layer activation
     *
     * @return the values of the output layer activation
     */
    public double[] getOutputLayerActivations() {
        double[] values = new double[this.outputLayer.getLayerSize()];
        for (int i = 0; i < values.length; i++) {
            values[i] = this.outputLayer.getNodes()[i].getActivation();
        }
        return values;
    }

    /**
     * Determines the index of the class with the highest value
     *
     * @return the index of the class with the highest value
     */
    public int getDecision() {
        double[] values = getOutputLayerActivations();
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

    /**
     * Only used for making generating random numbers in a specific range easier
     *
     * @param lowerLimit the lower limit of the random number
     * @param upperLimit the upper limit of the random number
     * @return the random number
     * @// TODO: 28.03.2021 put this in another class
     */
    private double randomValue(double lowerLimit, double upperLimit) {
        return Math.random() * (upperLimit - lowerLimit) + lowerLimit;
    }

    /**
     * Sets the learning rate of the network
     *
     * @param learningRate the learning rate the network should use. Too high values can result in the network adjusting the weight to violently and overshooting its target. Too low values can result in slow learning
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }


    /**
     * Adds a layer to the network
     *
     * @param layer the layer that should be added
     */
    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    /**
     * @return whether the network uses biases
     */
    public boolean isUseBias() {
        return useBias;
    }

    /**
     * Used to make the network use biases or not
     *
     * @param useBias whether the network should use biases
     */
    public void setUseBias(boolean useBias) {
        this.useBias = useBias;
    }

    /**
     * Applies all the accumulated changes to all weights an biases in the network
     */
    public void applyChanges() {
        for (int i = 1; i < this.layers.size(); i++) {
            Layer layer = this.layers.get(i);
            for (Node node :
                    layer.getNodes()) {
                for (Weight weight :
                        node.getWeights()) {
                    weight.applyChanges(this.learningRate);
                    weight.clearChanges();
                }
                if (this.useBias) {
                    node.getBias().applyChanges(this.learningRate);
                    node.getBias().clearChanges();
                }
            }
        }
    }

    /**
     * Sets the cost function of the neural network
     *
     * @param costFunction the cost function that should be used
     */
    public void setCostFunction(CostFunctions costFunction) {
        this.costFunction = costFunction;
    }
}