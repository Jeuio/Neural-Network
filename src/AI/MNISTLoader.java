package AI;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class MNISTLoader {
    private ArrayList<double[]> data = new ArrayList<>(); //Stores the extracted data of the dataset
    private ArrayList<double[]> labels = new ArrayList<>(); //Stores the extracted labels of the dataset

    /**
     * Method to extract all the data from the database
     * @throws FileNotFoundException is thrown when the path is not specified correctly
     */
    public void extract() throws FileNotFoundException {
        String inputPath = "src\\AI\\Database\\mnist_train.csv"; //Location of the csv file
        ArrayList<Integer> integers = new ArrayList<>();
        Scanner scanner = new Scanner(new File(inputPath));
        while (scanner.hasNextLine()) {
            String[] content = scanner.nextLine().split(",");
            for (String s :
                    content) {
                integers.add(Integer.parseInt(s));
            }
        }
        scanner.close();

        for (int i = 0; i < integers.size() / 785; i++) {
            double[] label = new double[10];
            double[] data = new double[784];
            for (int j = 0; j < 785; j++) {
                double value = integers.get(i * 785 + j);
                if (j == 0) {
                    label[(int)value] = 1;
                } else {
                    data[j - 1] = value / 255d;
                }
            }
            this.labels.add(label);
            this.data.add(data);
        }
    }

    /**
     * @return the data of the dataset
     */
    public ArrayList<double[]> getData() {
        return this.data;
    }

    /**
     * @return the labels of the dataset
     */
    public ArrayList<double[]> getLabels() {
        return labels;
    }
}