import AI.AiBuilder;
import AI.SquishificationFunction;
import AI.PictureToData;

import java.io.FileNotFoundException;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws FileNotFoundException {

        AiBuilder builder = new AiBuilder(3, 256, 20, 10);
        builder.setSquishificationFunction(SquishificationFunction.RELU);
        builder.build();

        try {
            builder.assignValues(PictureToData.picuteToData("src\\placeholder.jpg"));
        } catch (IOException e) {
            System.err.println("The specified file could not be found");
        }

        builder.calculateValues();
        for (float f:
             builder.getOutputLayerValues()) {
            System.out.println(f);
        }
    }
}
