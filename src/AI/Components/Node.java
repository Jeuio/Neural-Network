package AI.Components;

public class Node {

    private Weight[] weights; //The weights influencing its output
    private double value = 0; //The value of the node
    private double activation = 0; //The value of the node after it was passed through the activation function of the layer
    private Bias bias; //The bias of the node
    private byte layerPosition = 0; //The index of the layer the node is in
    private Node[] inputNodes; //The input nodes of the node
    private double cost = 0; //The cost of the node
    private int indexInLayer; //The position of the node in the layer

    /**
     * @return the weights influencing its output
     */
    public Weight[] getWeights() {
        return weights;
    }

    /**
     * Sets the weights of this neuron
     * @param weights the weights of this neuron
     */
    public void setWeights(Weight[] weights) {
        this.weights = weights;
    }

    /**
     * @return the value of the neuron
     */
    public double getValue() {
        return this.value;
    }

    /**
     * Sets the value of the neuron
     * @param value the value that should be set
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * @return the value of the neuron after it was passed through the activation function
     */
    public double getActivation() {
        return activation;
    }

    /**
     * Sets the activation of the neuron
     * @param value the value of the activation
     */
    public void setActivation(double value) {
        this.activation = value;
    }

    /**
     * @return the bias of the neuron
     */
    public Bias getBias() {
        return bias;
    }

    /**
     * Sets the bias of the neuron
     * @param bias the bias that should be set
     */
    public void setBias(Bias bias) {
        this.bias = bias;
    }

    /**
     * @return the index of the layer the node is in
     */
    public byte getLayerPosition() {
        return layerPosition;
    }

    /**
     * Sets the index of the layer the node is in
     * @param layerPosition the index of the layer the node is in
     */
    public void setLayerPosition(byte layerPosition) {
        this.layerPosition = layerPosition;
    }

    /**
     * @return the input nodes of the node
     */
    public Node[] getInputNodes() {
        return this.inputNodes;
    }

    /**
     * Sets the input nodes of the node
     * @param node the input nodes of the neuron
     */
    public void setInputNodes(Node[] node) {
        this.inputNodes = node;
    }

    /**
     * @return the cost of the node
     */
    public double getCost() {
        return cost;
    }

    /**
     * Sets the cost of the node
     * @param cost the cost the node should have
     */
    public void setCost(double cost) {
        this.cost = cost;
    }

    /**
     * @return the position of the node in the layer
     */
    public int getIndexInLayer() {
        return indexInLayer;
    }

    /**
     * Sets the position of the node in the layer
     * @param indexInLayer the position of the node in the layer
     */
    public void setIndexInLayer(int indexInLayer) {
        this.indexInLayer = indexInLayer;
    }
}
