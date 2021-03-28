package AI.Learning;

import java.util.ArrayList;

/**
 * This class handles the creating of mini batches
 */
public class MinibatchCreator {

    private ArrayList<double[]> data; //The data that should be distributed between the batches
    private ArrayList<double[]> labels; //The labels that should be distributed between the batches
    private Minibatch[] minibatches; //All the mini batches that have been created

    /**
     * This method tries to create mini batches of a specified size. When the sizes can't be met, it will try to distribute
     * the remaining data evenly between the batches, so no data will be lost
     *
     * @param prefferedBatchSize the preferred size of the batches
     */
    public void createMiniBatches(int prefferedBatchSize) {
        assert data.size() == labels.size() : "data size does not match label size";
        int dataSize = data.size();
        int remainder = data.size() % prefferedBatchSize;
        int totalMinibatches = (dataSize - remainder) / prefferedBatchSize;
        ArrayList<double[]> d = (ArrayList<double[]>) data.clone();
        ArrayList<double[]> l = (ArrayList<double[]>) labels.clone();

        this.minibatches = new Minibatch[totalMinibatches];
        for (int i = 0; i < totalMinibatches; i++) {
            Minibatch minibatch = new Minibatch();
            ArrayList<double[]> minibatchData = new ArrayList<>();
            ArrayList<double[]> minibatchLabels = new ArrayList<>();
            int batchSize = prefferedBatchSize;
            if (remainder > 0) {
                batchSize++;
                remainder--;
            }
            for (int j = 0; j < batchSize; j++) {
                int r = (int)(Math.random() * d.size());
                minibatchData.add(d.get(r));
                minibatchLabels.add(l.get(r));
                d.remove(r);
                l.remove(r);
                d.trimToSize();
                l.trimToSize();
            }
            minibatch.setData(minibatchData);
            minibatch.setLabels(minibatchLabels);

            this.minibatches[i] = minibatch;
        }
    }

    /**
     * @return the data this class is storing
     */
    public ArrayList<double[]> getData() {
        return data;
    }

    /**
     * Sets the data this class should work with
     * @param data the data this class should work with
     */
    public void setData(ArrayList<double[]> data) {
        this.data = data;
    }

    /**
     * @return the labels this class is storing
     */
    public ArrayList<double[]> getLabels() {
        return labels;
    }

    /**
     * Sets the labels this class should work with
     * @param labels the labels this class should work with
     */
    public void setLabels(ArrayList<double[]> labels) {
        this.labels = labels;
    }

    /**
     * @return the mini batches that have been created. Should only be invoked after createMiniBatches was invoked
     */
    public Minibatch[] getMinibatches() {
        return this.minibatches;
    }
}
