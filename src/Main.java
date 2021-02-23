import AI.AiBuilder;
import AI.MNISTLoader;
import AI.SquishificationFunction;
import java.io.FileNotFoundException;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        AiBuilder builder = new AiBuilder(3, 784, 20, 10);
        builder.setSquishificationFunction(SquishificationFunction.RELU);
        builder.setLearningRate((float) 0.01);
        builder.build();
        MNISTLoader loader = new MNISTLoader();
        loader.extract();
        builder.learn(loader.getData(), loader.getLabels());
    }
}
