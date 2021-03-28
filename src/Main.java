import AI.AiBuilder;
import AI.Components.Layer;
import AI.Cost.CostFunctions;
import AI.GUI.GUI;
import AI.LayerType;
import AI.ActivationFunction;
import AI.MNISTLoader;
import java.io.FileNotFoundException;

/**
 * This is the main class. Its primary focus is to create a neural network that can recognize handwritten digits
 */
public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        new GUI();
        AiBuilder builder = new AiBuilder();
        Layer inputLayer = new Layer(784, 0, ActivationFunction.NONE, LayerType.INPUT);
        Layer hiddenLayer = new Layer(100, 1, ActivationFunction.SIGMOID, LayerType.HIDDEN);
        Layer outputLayer = new Layer(10, 2, ActivationFunction.SOFTMAX, LayerType.OUTPUT);

        builder.addLayer(inputLayer);
        builder.addLayer(hiddenLayer);
        builder.addLayer(outputLayer);

        builder.setUseBias(true);
        builder.setLearningRate(0.001f);
        builder.setCostFunction(CostFunctions.CROSS_ENTROPY);

        builder.build();

        MNISTLoader loader = new MNISTLoader();
        loader.extract();

        builder.learn(1000, 500, "src\\AI\\Progress\\progress.txt", loader.getData(), loader.getLabels());
    }
}
