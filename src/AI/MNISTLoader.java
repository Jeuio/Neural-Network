package AI;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class MNISTLoader {
    private ArrayList<double[]> data = new ArrayList<>();
    private ArrayList<double[]> labels = new ArrayList<>();

    /**
     * Method to extract all the data from the database
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
        scanner.close();

        //Creates a new entry in data for each 785 entries in the list
        double[] doubleArray = new double[784];
        int currentIndex = 0;
        for (int i :
                integers) {
            if (currentIndex == 0) {
                double[] label = new double[10];
                label[i] = 1;
                labels.add(label);
            } else {
                doubleArray[currentIndex - 1] = 1.0f / 255.0f * (double)i;
            }
            currentIndex++;
            if (currentIndex == 785) {
                currentIndex = 0;
                data.add(doubleArray);
            }
        }
    }

    public ArrayList<double[]> getData() {
        return this.data;
    }

    public ArrayList<double[]> getLabels() {
        return labels;
    }
}