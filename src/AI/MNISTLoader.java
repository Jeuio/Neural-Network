package AI;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class MNISTLoader {
    private ArrayList<Integer> data = new ArrayList<>();

    public void extractor() throws FileNotFoundException {
        String inputPath = "src\\TrainImages\\mnist_train.csv";
        Scanner scanner = new Scanner(new File(inputPath));
        while (scanner.hasNext()) {
            String[] content = scanner.next().split(",");
            for (String s :
                    content) {
                data.add(Integer.parseInt(s));
            }
        }
    }

    public ArrayList<Integer> getData() {
        return data;
    }
}