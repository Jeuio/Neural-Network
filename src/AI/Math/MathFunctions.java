package AI.Math;
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
}