package AI.Learning;

import java.util.ArrayList;

public class MinibatchCreator {

    private ArrayList<double[]> data;
    private ArrayList<double[]> labels;
    private Minibatch[] minibatches;

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

            /*
            System.out.println("minibatch " + i);
            for (int j = 0; j < minibatch.getData().size(); j++) {
                System.out.println("dataset " + j);
                for (int k = 0; k < minibatch.getData().get(j).length; k++) {
                    System.out.println(minibatch.getData().get(j)[k]);
                }
                System.out.println("");
            }
            System.out.println("");

             */


            this.minibatches[i] = minibatch;
        }
    }

    public ArrayList<double[]> getData() {
        return data;
    }

    public void setData(ArrayList<double[]> data) {
        this.data = data;
    }

    public ArrayList<double[]> getLabels() {
        return labels;
    }

    public void setLabels(ArrayList<double[]> labels) {
        this.labels = labels;
    }

    public Minibatch[] getMinibatches() {
        return minibatches;
    }
}
