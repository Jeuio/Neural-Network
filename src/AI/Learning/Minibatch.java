package AI.Learning;

import java.util.ArrayList;

/**
 * This class stores the training data and labels of a mini batch
 */
public class Minibatch {

    private ArrayList<double[]> data; //The data of the mini batch
    private ArrayList<double[]> labels; //The labels of the mini batch

    /**
     * @return the data of the mini batch
     */
    public ArrayList<double[]> getData() {
        return this.data;
    }

    /**
     * Sets the data of the mini batch
     * @param data the data the mini batch should have
     */
    public void setData(ArrayList<double[]> data) {
        this.data = data;
    }

    /**
     * @return the labels of the mini batch
     */
    public ArrayList<double[]> getLabels() {
        return this.labels;
    }

    /**
     * Sets the labels of the mini batch
     * @param labels the labels the mini batch should have
     */
    public void setLabels(ArrayList<double[]> labels) {
        this.labels = labels;
    }
}
