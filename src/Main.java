import AI.AiBuilder;
import AI.SquishificationFunction;
import java.io.FileNotFoundException;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        AiBuilder builder = new AiBuilder(3, 256, 20, 10);
        builder.setSquishificationFunction(SquishificationFunction.RELU);
        builder.build();

        builder.calculateValues();
        for (float f:
             builder.getOutputLayerValues()) {
            System.out.println(f);
        }
    }
}
