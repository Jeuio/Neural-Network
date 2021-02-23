package AI.Math;

public class MathFunctions {

    //Functions to refine the values of nodes (https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#id10)

    public static float relu(float value) {
        return Math.max(0, value);
    }

    public static float leakyRelu(float value, float alpha, boolean derivative) {
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
}