package AI.Components;

public class Node {

    private double[] weights;
    private double value = 0;
    private double bias = 0;
    private byte layerPosition = 0;
    private Node[] inputNodes;
    private double cost = 0;

    //Getter and setter for all variables

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public byte getLayerPosition() {
        return layerPosition;
    }

    public void setLayerPosition(byte layerPosition) {
        this.layerPosition = layerPosition;
    }

    public Node[] getInputNodes() {
        return this.inputNodes;
    }

    public void setInputNodes(Node[] node) {
        this.inputNodes = node;
    }

    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }
}
