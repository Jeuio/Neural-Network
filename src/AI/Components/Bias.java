package AI.Components;

/**
 * Quite similar to the weight class
 * @see Weight
 *
 * @// TODO: 28.03.2021 create a super class for the Weight and Bias class due to high similarity
 *
 * This class handles the creation and modification of biases
 */
public class Bias {

    private double changeSum = 0; //Used for storing the sum of all added changes
    private int totalChanges = 0; //Used for keeping count of the number of changes
    private double value; //The value of the Bias

    /**
     * Constructor for creating a Bias
     * @param value initial value of the bias
     */
    public Bias(double value) {
        this.value = value;
    }

    /**
     * Used for accumulating changes
     * @param value value of the change that should be added
     */
    public void addChange(double value) {
        this.changeSum += value;
        this.totalChanges++;
    }

    /**
     * Resets the accumulated changes. Should be used after applyChanges has been invoked
     */
    public void clearChanges() {
        this.changeSum = 0;
        this.totalChanges = 0;
    }


    /**
     * Applies all the changes that have been set
     * @param learningRate the learning rate that should be used
     */
    public void applyChanges(double learningRate) {
        this.value -= changeSum / this.totalChanges * learningRate;
    }

    /**
     * @return the value of the bias
     */
    public double getValue() {
        return this.value;
    }
}
