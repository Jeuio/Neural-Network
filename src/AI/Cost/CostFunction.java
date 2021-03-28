package AI.Cost;

import AI.Components.Node;

/**
 * This class stores the cost functions to determine the cost of the network
 *
 * @// TODO: 28.03.2021 add more functions
 */
public class CostFunction {

    /**
     * Better for classification problems
     * Calculated the cost of the network. Should be used with values that can be interpreted as percentages
     *
     * @param outputNodes   the nodes in the output layer
     * @param desiredValues the values the network should achieve
     * @return the calculated cost
     */
    public static double crossEntropy(Node[] outputNodes, double[] desiredValues) {
        int rightAnswerIndex = 0;

        for (double desiredValue :
                desiredValues) {
            if (desiredValue == 1) {
                rightAnswerIndex = 1;
                break;
            }
        }
        return -Math.log(outputNodes[rightAnswerIndex].getActivation());
    }

    /**
     * Calculates the derivative of the cross entropy function. Can only be used with values that can be interpreted as percentages
     *
     * @param outputNodes   the nodes in the output layer
     * @param desiredValues the values the network should achieve
     * @return the calculated derivative of the cost function
     */
    public static double[] derivativeCrossEntropy(Node[] outputNodes, double[] desiredValues) {
        double[] derivatives = new double[outputNodes.length];

        int rightAnswerIndex = 0;

        for (int i = 0; i < desiredValues.length; i++) {
            if (desiredValues[i] == 1) {
                rightAnswerIndex = i;
            }
        }

        for (int i = 0; i < outputNodes.length; i++) {
            derivatives[i] = 0;
            if (i == rightAnswerIndex) {
                derivatives[i] += outputNodes[i].getActivation() - 1;
            } else {
                derivatives[i] += outputNodes[i].getActivation();
            }
        }

        return derivatives;
    }

    /**
     * Better for linear regression problems
     *
     * @param predictedValue predicted value of a node
     * @param realValue      value the node should achieve
     * @return the cost of the network
     */
    public static double meanSquaredError(double predictedValue, double realValue, boolean derivative) {
        return Math.pow(realValue - predictedValue, 2);
    }
}