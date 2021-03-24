package AI.Cost;

import AI.Components.Node;

public class CostFunction {

    /**
     * Better for classification problems.
     *
     * @param outputNodes   nodes in the outputlayer
     * @param desiredValues value the node should achieve
     * @return cost
     */
    public static double crossEntropy(Node[] outputNodes, double[] desiredValues) {
        //return realValue * Math.log(predictedValue) + (1 - realValue) * Math.log(1 - predictedValue);
        double sum = 0;
        for (int i = 0; i < outputNodes.length; i++) {
            sum += desiredValues[i] + Math.log(outputNodes[i].getActivation());
        }
        return -sum;
    }

    public static double[] derivativeCrossEntropy(Node[] outputNodes, double[] desiredValues) {
        double[] derivatives = new double[outputNodes.length];
        for (int i = 0; i < derivatives.length; i++) {
            derivatives[i] = (outputNodes[i].getActivation() - desiredValues[i]) / ((outputNodes[i].getActivation() * (1 - outputNodes[i].getActivation())) + 0.0000000000000000000000000000000000000001);
        }
        return derivatives;
    }

    /**
     * Better for linear regression.
     *
     * @param predictedValue predicted value of a node
     * @param realValue      value the node should achieve
     * @return cost
     */
    public static double meanSquaredError(double predictedValue, double realValue, boolean derivative) {
        return Math.pow(realValue - predictedValue, 2);
    }
}
