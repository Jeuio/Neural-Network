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

public class AiBuilder {

    private Layer inputLayer; //Array of all nodes in the input layer
    private Layer outputLayer; //Array of all nodes in the output layer
    private double learningRate; //Learning rate of the network
    private boolean useBias = true;
    private CostFunctions costFunction;
    private ArrayList<Layer> layers = new ArrayList<>();

    /**
     * Links all the nodes in the network by calling the link method
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
                    newNode.setBias(new Bias(randomValue(-Math.sqrt(this.layers.get(i - 1).getLayerSize()), Math.sqrt(this.layers.get(i - 1).getLayerSize())))); //Assigns a random bias to the node
                }
                nodeArray[j] = newNode; //Adds the new node to the node array
            }
            this.layers.get(i).setNodes(nodeArray); //Adds the node array to the layer list
        }
        inputLayer = this.layers.get(0); //Assigns the first layer to "inputLayer"
        outputLayer = this.layers.get(this.layers.size() - 1); //Assigns the last Layer to "outputLayer"

        for (Node node : //Goes through all nodes in the output layer
                outputLayer.getNodes()) {
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
            n.setInputNodes(this.layers.get(n.getLayerPosition() - 1).getNodes()); //Assigns all the nodes from the previous layer as input nodes
            for (Node node : //Does the same for all nodes in the previous layer
                    this.layers.get(n.getLayerPosition() - 1).getNodes()) {
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
        if (values.length != this.layers.get(0).getLayerSize()) { //Checks whether the length of the values array is equal to the amount of nodes in the input layer and throws an exception otherwise
            throw new IllegalArgumentException();
        }
        for (int i = 0; i < values.length; i++) {
            inputLayer.getNodes()[i].setValue(values[i]); //Assign the value of the value array to the nodes
        }
    }

    /**
     * Propagates all the values in the neural network by calling the propagate method
     */
    public void calculateValues() {
        for (Node node :
                outputLayer.getNodes()) {
            propagate(node);
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
            Weight[] weights = node.getWeights(); //Array of all the weights connected to the node
            for (int i = 0; i < node.getInputNodes().length; i++) { //Iterate through all the input nodes of the node
                value += propagate(node.getInputNodes()[i]) * weights[i].getValue(); //Calculate the activation of the node
            }
            if (this.useBias) {
                value += node.getBias().getValue();
            }

            node.setValue(value);

            if (this.layers.get(node.getLayerPosition()).getActivationFunction() == ActivationFunction.RELU) {
                value = MathFunctions.relu(value);
            } else if (this.layers.get(node.getLayerPosition()).getActivationFunction() == ActivationFunction.LEAKYRELU) {
                value = MathFunctions.leakyRelu(value, -0.01, false);
            } else if (this.layers.get(node.getLayerPosition()).getActivationFunction() == ActivationFunction.SIGMOID) {
                value = MathFunctions.sigmoid(value, false);
            } else if (this.layers.get(node.getLayerPosition()).getActivationFunction() == ActivationFunction.SOFTMAX) {
                value = MathFunctions.softmax(node.getIndexInLayer(), this.layers.get(node.getLayerPosition()).getNodes(), false);
            }

            node.setActivation(value);
            return value;
        } else {
            return node.getActivation();
        }
    }

    /**
     * Method that lets the network learn
     *
     * @param data           an array of data to learn
     * @param expectedOutput the expected output for each data segment
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
        int totalNodes = 0;
        for (int i = 1; i < this.layers.size(); i++) {
            totalNodes += this.layers.get(i).getLayerSize();
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
                    calculateValues();
                    calculateCostForNodes(outputLayer.getLayerPosition(), minibatch.getLabels().get(k));

                    sumOfCost = sumOfCost.add(networkCost());

                    int biggestValueIndex = 0;
                    for (int l = 0; l < minibatch.getLabels().get(k).length; l++) {
                        if (minibatch.getLabels().get(k)[l] > minibatch.getLabels().get(k)[biggestValueIndex]) {
                            biggestValueIndex = l;
                        }
                    }
                    if (getDecision() == biggestValueIndex) {
                        correctPredictions++;
                    }
                    backpropagate(0);
                }
                applyChanges();
            }
            safeProgress(filePath);
            System.out.println("Epoch: " + i + "; Average cost of epoch: " + sumOfCost.divide(BigDecimal.valueOf(totalNodes), RoundingMode.FLOOR) + "; Percentage correct: " + (double)correctPredictions / (double)data.size() * 100 + "%; Calculation time: " + (System.currentTimeMillis() - startTime) / 1000 + "s");

            /*
            for (int j = 0; j < data.size(); j++) {
                assignValues(data.get(j));
                calculateValues();
                calculateCostForNodes(outputLayer.getLayerPosition(), expectedOutput.get(j));

                sumOfCost = sumOfCost.add(networkCost());

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
            applyChanges();
            safeProgress(filePath);
            System.out.println("Epoch: " + i + "; Average cost of epoch: " + sumOfCost.divide(BigDecimal.valueOf(totalNodes), RoundingMode.FLOOR) + "; Percentage correct: " + (double)correctPredictions / (double)data.size() * 100 + "%; Calculation time: " + (System.currentTimeMillis() - startTime) / 1000 + "s");

             */
        }
    }

    private BigDecimal networkCost() {
        BigDecimal sumOfCost = new BigDecimal(0);
        for (int i = 1; i < this.layers.size(); i++) {
            Layer layer = this.layers.get(i);
            for (int j = 0; j < layer.getNodes().length; j++) {
                Node node = layer.getNodes()[j];
                sumOfCost = sumOfCost.add(BigDecimal.valueOf(node.getCost()));
            }
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
                        writer.write("_"); //Symbol for separating weights for layers
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
     * Recursive function that applies the backpropagation algorithm
     * Geht durch alle Nodes in der output layer und berechnet die cost.
     * Durch jede node der vorherigen layer iterieren:
     *
     * error outputlayer = derivat der costfunktion in respect to aktivierung der node in der outputlayer * derivat der aktivierungsfunktion mit dem wert des neurons
     * error hiddenlayer = eine node nehmen, (die summe der cost der node, mit dem es verbunden ist * wert des weights auf dieser strecke) * derivat der aktivierungsfunktion mit dem wert des neurons
     *
     * gradient weight = aktivierung des linken neurons * die cost des rechten neurons
     * gradient bias = cost des neurons
     *
     * Diese werte für jedes weight speichern.
     * Diese werte nach jeder epoche aktualisieren
     *
     * @param layer the layer the weights are adjusted at
     * @implNote calculateCostForNodes has to be called to update the cost of the nodes
     */
    private void backpropagate(int layer) {
        if (layer < this.layers.size() - 1) {
            for (int rightNodeIndex = 0; rightNodeIndex < this.layers.get(layer + 1).getLayerSize(); rightNodeIndex++) {
                Node rightNode = this.layers.get(layer + 1).getNodes()[rightNodeIndex];
                Weight[] weights = rightNode.getWeights();
                for (int leftNodeIndex = 0; leftNodeIndex < this.layers.get(layer).getLayerSize(); leftNodeIndex++) {
                    Node leftNode = this.layers.get(layer).getNodes()[leftNodeIndex];
                    weights[leftNodeIndex].addChange(leftNode.getActivation() * rightNode.getCost()); //adding the gradient of the cost function for the weight
                }
                rightNode.setWeights(weights);
                if (this.useBias) {
                    rightNode.getBias().addChange(rightNode.getCost()); //adding the gradient of the cost function for the bias
                }
            }
            backpropagate(layer + 1);
        }
    }

    /**
     * Method that calculates the cost of the output layer nodes
     *
     * @param layer          the layer of which the cost of the nodes is calculated
     * @param expectedOutput an array of the expected output. Size has to match the number of output nodes
     */
    private void calculateCostForNodes(int layer, double[] expectedOutput) {
        for (int i = 0; i < this.layers.get(layer).getLayerSize(); i++) { //Iterates through all nodes of the layer
            Node node = this.layers.get(layer).getNodes()[i];
            double cost;

            if (this.costFunction == CostFunctions.CROSS_ENTROPY) {
                cost = CostFunction.derivativeCrossEntropy(this.layers.get(layer).getNodes(), expectedOutput)[node.getIndexInLayer()];
            } else {
                cost = node.getActivation() - expectedOutput[i];
            }

            if (this.layers.get(layer).getActivationFunction() == ActivationFunction.SIGMOID) {
                cost *= MathFunctions.sigmoid(node.getActivation(), true);
            } else if (this.layers.get(layer).getActivationFunction() == ActivationFunction.SOFTMAX) {
                cost *= MathFunctions.softmax(node.getIndexInLayer(), this.layers.get(node.getLayerPosition()).getNodes(), true);
            }
            node.setCost(cost); //Sets the cost of the node
        }
        if (layer > 1) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer); //Calls the other method to calculate the cost for the other layers
        }
    }

    /**
     * Recursive methos that calculate the cost of nodes in a specified layer
     *
     * @param layer the layer of which the cost of the nodes is calculated
     */
    private void calculateCostForNodes(int layer) {
        for (int i = 0; i < this.layers.get(layer - 1).getLayerSize(); i++) {
            Node node = this.layers.get(layer - 1).getNodes()[i];
            double sum = 0; //Sum of the cost
            for (int j = 0; j < layers.get(layer).getLayerSize(); j++) { //Iterated through all nodes of the layer
                Weight[] weights = this.layers.get(layer).getNodes()[j].getWeights(); //Weights of the node
                sum += layers.get(layer).getNodes()[j].getCost() * weights[j].getValue(); //Adds the cost of the node to the sum
            }
            double cost = sum;
            if (this.layers.get(layer - 1).getActivationFunction() == ActivationFunction.SIGMOID) {
                cost *= MathFunctions.sigmoid(node.getActivation(), true);
            } else if (this.layers.get(layer - 1).getActivationFunction() == ActivationFunction.SOFTMAX) {
                cost *= MathFunctions.softmax(node.getIndexInLayer(), this.layers.get(node.getLayerPosition()).getNodes(), true);
            }
            node.setCost(cost); //Sets the cost of the node
        }
        if (layer > 2) { //Checks whether the input layer has been reached
            calculateCostForNodes(layer - 1); //Call the method to calculate the cost of the previous layer
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

    private double randomValue(double lowerLimit, double upperLimit) {
        return Math.random() * (upperLimit - lowerLimit + 1) + lowerLimit;
    }

    /**
     * Sets the learning rate of the network.
     *
     * @param learningRate the learning rate the network should use. Too high values can result in the network adjusting the weight to violently and overshooting its target. Too low values can result in a too small learning effect.
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }

    public boolean isUseBias() {
        return useBias;
    }

    public void setUseBias(boolean useBias) {
        this.useBias = useBias;
    }

    public void applyChanges() {
        for (int i = 1; i < this.layers.size(); i++) {
            Layer layer =  this.layers.get(i);
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

    public void setCostFunction(CostFunctions costFunction) {
        this.costFunction = costFunction;
    }
}