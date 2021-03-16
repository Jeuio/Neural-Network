import AI.AiBuilder;
import AI.Components.Layer;
import AI.Cost.CostFunctions;
import AI.LayerType;
import AI.ActivationFunction;
import AI.MNISTLoader;

import java.io.FileNotFoundException;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        AiBuilder builder = new AiBuilder();
        Layer inputLayer = new Layer(784, 0, ActivationFunction.NONE, LayerType.INPUT);
        Layer hiddenLayer = new Layer(32, 1, ActivationFunction.SIGMOID, LayerType.HIDDEN);
        Layer outputLayer = new Layer(10, 2, ActivationFunction.SOFTMAX, LayerType.OUTPUT);
        /*
        Layer inputLayer = new Layer(2, 0, ActivationFunction.NONE, LayerType.INPUT);
        Layer hiddenLayer = new Layer(6, 1, ActivationFunction.SIGMOID, LayerType.HIDDEN);
        Layer outputLayer = new Layer(2, 2, ActivationFunction.SIGMOID, LayerType.OUTPUT);
         */
        builder.addLayer(inputLayer);
        builder.addLayer(hiddenLayer);
        builder.addLayer(outputLayer);
        builder.setUseBias(true);
        builder.setLearningRate(0.0001f);
        builder.setCostFunction(CostFunctions.CROSS_ENTROPY);
        builder.build();


        MNISTLoader loader = new MNISTLoader();
        loader.extract();

        /*
        ArrayList<double[]> data = new ArrayList<>();
        ArrayList<double[]> labels = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            double[] d = new double[2];
            double[] l = new double[2];
            double x = Math.floor(Math.random() * 2);
            double y = Math.floor(Math.random() * 2);

            if (x == 1 && y == 1 || x == 0 && y == 0) {
                l[0] = 1;
            } else if (x == 0 && y == 1 || x == 1 && y == 0) {
                l[1] = 1;
            }
            d[0] = x;
            d[1] = y;
            data.add(d);
            labels.add(l);
        }
        */

        builder.learn(1000, "src\\AI\\Progress\\progress.txt", loader.getData(), loader.getLabels());
    }
}
