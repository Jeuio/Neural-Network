package AI.Components;

import AI.ActivationFunction;
import AI.LayerType;

public class Layer {

    private int layerPosition;
    private ActivationFunction activationFunction;
    private LayerType layerType;
    private int layerSize;
    private Node[] nodes;

    public Layer(int layerSize, int layerPosition, ActivationFunction activationFunction, LayerType layerType) {
        this.layerSize = layerSize;
        this.layerPosition = layerPosition;
        this.activationFunction = activationFunction;
        this.layerType = layerType;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public int getLayerPosition() {
        return layerPosition;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public LayerType getLayerType() {
        return layerType;
    }

    public Node[] getNodes() {
        return nodes;
    }

    public void setNodes(Node[] nodes) {
        this.nodes = nodes;
    }
}
