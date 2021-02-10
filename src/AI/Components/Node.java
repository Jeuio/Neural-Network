package AI.Components;

public class Node {

    private float weight;
    private float value = 0;
    private float bias = 0;
    private byte layerPosition = 0;
    private Node[] inputNodes;

    //Getter and setter for all variables

    public float getWeight() {
        return weight;
    }

    public void setWeight(float weight) {
        this.weight = weight;
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
}
