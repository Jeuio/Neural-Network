package AI.Learning;

import java.util.ArrayList;

public class Minibatch {

    private ArrayList<double[]> data;
    private ArrayList<double[]> labels;

    public ArrayList<double[]> getData() {
        return this.data;
    }

    public void setData(ArrayList<double[]> data) {
        this.data = data;
    }

    public ArrayList<double[]> getLabels() {
        return this.labels;
    }

    public void setLabels(ArrayList<double[]> labels) {
        this.labels = labels;
    }
}
