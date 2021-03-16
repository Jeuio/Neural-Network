package AI.Components;

public class Node {

    private Weight[] weights;
    private double value = 0;
    private double activation = 0;
    private Bias bias;
    private byte layerPosition = 0;
    private Node[] inputNodes;
    private double cost = 0;
    private int indexInLayer;

    //Getter and setter for all variables

    public Weight[] getWeights() {
        return weights;
    }

    public void setWeights(Weight[] weights) {
        this.weights = weights;
    }

    public double getValue() {
        return this.value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getActivation() {
        return activation;
    }

    public void setActivation(double value) {
        this.activation = value;
    }

    public Bias getBias() {
        return bias;
    }

    public void setBias(Bias bias) {
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

    public int getIndexInLayer() {
        return indexInLayer;
    }

    public void setIndexInLayer(int indexInLayer) {
        this.indexInLayer = indexInLayer;
    }
}
