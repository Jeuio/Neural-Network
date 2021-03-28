package AI.Math;

import AI.Components.Node;

/**
 * This class stores activation functions that can be used to "activate" nodes
 *
 * @// TODO: 28.03.2021 rename this class
 */
public class MathFunctions {

    /**
     * Calculates the activation of a node. This function can return values from 0 to infinity
     *
     * @param value      the value of a node
     * @param derivative whether the derivative should be calculated instead
     * @return the activation of the node
     */
    public static double relu(double value, boolean derivative) {
        if (!derivative) {
            return Math.max(0, value);
        } else {
            if (value > 0) {
                return 1;
            } else {
                return 0;
            }
        }
    }

    /**
     * Calculates the activation of a node. This function can return values from -infinity to infinity.
     * This function is similar to the relu function. This function has a gradient in the negative spectrum
     *
     * @param value      the value of a node
     * @param alpha      defines how steep the gradient is in the negative spectrum
     * @param derivative whether the derivative should be calculated instead
     * @return the activation of the node
     */
    public static double leakyRelu(double value, double alpha, boolean derivative) {
        if (!derivative) {
            if (value > 0) {
                return value;
            } else {
                return alpha * value;
            }
        } else { //Return the derivative
            if (value >= 0) {
                return 1;
            } else {
                return Math.abs(alpha);
            }
        }
    }

    /**
     * Commonly used function to calculate the activation of a node. This function can return values from 0 to 1.
     * This function in non-linear, which makes it very well suited for gradient descent
     *
     * @param value      the value of the node
     * @param derivative whether the derivative should be calculated instead
     * @return the activation of the node
     */
    public static double sigmoid(double value, boolean derivative) {
        if (!derivative) {
            return (1d / (1 + (Math.pow(Math.E, (-value)))));
        } else {
            return sigmoid(value, false) * (1 - sigmoid(value, false));
        }
    }

    /**
     * Calculates the activation of a node. This function returns a value that can be interpreted as a percentage. The sum of all outputs of this function is always 1.
     * This function return values from 0 to 1
     *
     * @param index      the index of the value of node the percentage should be calculated from
     * @param nodeValues the values of the nodes
     * @param derivative whether the derivative should be calculated instead
     * @return a percentage
     */
    public static double softmax(int index, double[] nodeValues, boolean derivative) {
        if (!derivative) {
            double sum = 0;
            for (double value :
                    nodeValues) {
                sum += Math.pow(Math.E, value);
                if (Double.isNaN(sum)) {
                    return 1;
                }
            }
            return Math.pow(Math.E, nodeValues[index]) / sum;
        } else {
            return softmax(index, nodeValues, false) * (1 - softmax(index, nodeValues, false));
        }
    }

    /**
     * Calculates the activation of a node. This function returns a value that can be interpreted as a percentage. The sum of all outputs of this function is always 1.
     * This function return values from 0 to 1
     *
     * @param index      the index of the node the percentage should be calculated from
     * @param nodes      the nodes
     * @param derivative whether the derivative should be calculated instead
     * @return a percentage
     */
    public static double softmax(int index, Node[] nodes, boolean derivative) {
        if (!derivative) {
            double sum = 0;
            for (Node node :
                    nodes) {
                sum += Math.pow(Math.E, node.getValue());
            }
            return Math.pow(Math.E, nodes[index].getValue()) / sum;
        } else {
            return softmax(index, nodes, false) * (1 - softmax(index, nodes, false));
        }
    }
}