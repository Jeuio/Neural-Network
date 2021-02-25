import AI.AiBuilder;
import AI.MNISTLoader;
import AI.SquishificationFunction;
import java.io.FileNotFoundException;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        AiBuilder builder = new AiBuilder(3, true, 784, 100, 10);
        builder.setSquishificationFunction(SquishificationFunction.SIGMOID);
        builder.setLearningRate((float) 0.0001);
        builder.build();
        MNISTLoader loader = new MNISTLoader();
        loader.extract();
        builder.learn(1000, "src\\AI\\Progress\\progress.txt", loader.getData(), loader.getLabels());
    }
}
