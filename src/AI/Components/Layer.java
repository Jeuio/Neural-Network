package AI.Components;

import AI.ActivationFunction;
import AI.LayerType;

/**
 * This class handles the creation and modification of layers
 */
public class Layer {

    private final int layerPosition; //The position of the layer. Starting at "0" for the input layer
    private final ActivationFunction activationFunction; //The activation function of the layer
    private final LayerType layerType; //The type of the layer. Might be "input", "hidden", or "output"
    private final int layerSize; //The amount of nodes in this layer

    private Node[] nodes; //Stores the nodes of the layer

    /**
     * Constructor for creating a layer
     * @param layerSize the amount of nodes in this layer
     * @param layerPosition the position of the layer in the network
     * @param activationFunction the activation function that should be used for this layer
     * @param layerType the type of the layer
     */
    public Layer(int layerSize, int layerPosition, ActivationFunction activationFunction, LayerType layerType) {
        this.layerSize = layerSize;
        this.layerPosition = layerPosition;
        this.activationFunction = activationFunction;
        this.layerType = layerType;
    }

    /**
     * @return the amount of nodes in this layer
     */
    public int getLayerSize() {
        return layerSize;
    }

    /**
     * @return the position of the layer
     */
    public int getLayerPosition() {
        return layerPosition;
    }

    /**
     * @return the activation function of the layer
     */
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    /**
     * @return the type of the layer
     */
    public LayerType getLayerType() {
        return layerType;
    }

    /**
     * @return the nodes of this layer
     */
    public Node[] getNodes() {
        return nodes;
    }

    /**
     * Used to specify the nodes of this layer
     * @param nodes the nodes that should be set
     */
    public void setNodes(Node[] nodes) {
        this.nodes = nodes;
    }
}
