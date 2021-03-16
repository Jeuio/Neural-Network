package AI.Math;

import AI.Components.Node;

public class MathFunctions {

    //Functions to refine the values of nodes (https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#id10)

    public static double relu(double value) {
        return Math.max(0, value);
    }

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

    public static double sigmoid(double value, boolean derivative) {
        if (!derivative) {
            return (1d / (1 + Math.pow(Math.E, (-value))));
        } else {
            return sigmoid(value, false) * (1 - sigmoid(value, false));
        }
    }

    public static double softmax(int index, double[] nodeValues, boolean derivative) {
        if (!derivative) {
            double sum = 0;
            for (double value :
                    nodeValues) {
                sum += Math.pow(Math.E, value);
            }
            return Math.pow(Math.E, nodeValues[index]) / sum;
        } else {
            return softmax(index, nodeValues, false) * (1 - softmax(index, nodeValues, false));
        }
    }

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