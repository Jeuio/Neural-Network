package AI.Components;

public class Node {

    private float[] weights;
    private float value = 0;
    private float bias = 0;
    private byte layerPosition = 0;
    private Node[] inputNodes;
    private float cost = 0;

    //Getter and setter for all variables

    public float[] getWeights() {
        return weights;
    }

    public void setWeights(float[] weights) {
        this.weights = weights;
    }

    public float getValue() {
        return value;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
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

    public float getCost() {
        return cost;
    }

    public void setCost(float cost) {
        this.cost = cost;
    }
}
