package AI;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class MNISTLoader {
    private ArrayList<float[]> data = new ArrayList<>();
    private ArrayList<float[]> labels = new ArrayList<>();

    /**
     * Method to extract all the data from the database
     *
     * @throws FileNotFoundException
     */
    public void extract() throws FileNotFoundException {
        String inputPath = "src\\AI\\Database\\mnist_train.csv"; //Location of the csv file
        ArrayList<Integer> integers = new ArrayList<>();
        Scanner scanner = new Scanner(new File(inputPath));
        while (scanner.hasNext()) {
            String[] content = scanner.next().split(",");
            for (String s :
                    content) {
                integers.add(Integer.parseInt(s));
            }
        }

        //Creates a new entry in data for each 784 entries in the list
        float[] intArray = new float[784];
        int currentIndex = 0;
        for (int i :
                integers) {
            if (currentIndex == 0) {
                float[] label = new float[10];
                System.out.println(i);
                label[i] = 1;
                labels.add(label);
            } else {
                intArray[currentIndex - 1] = i;
            }
            currentIndex++;
            if (currentIndex == 785) {
                currentIndex = 0;
                data.add(intArray);
            }
        }
    }

    public ArrayList<float[]> getData() {
        return this.data;
    }

    public ArrayList<float[]> getLabels() {
        return labels;
    }
}