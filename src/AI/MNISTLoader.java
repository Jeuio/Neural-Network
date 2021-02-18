package AI;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class MNISTLoader {
    private ArrayList<int[]> data = new ArrayList<>();

    //Extracts the data of the MNIST database
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

        //Creates a new entry in data for each 785 entries in the list
        int[] intArray = new int[785];
        int currentIndex = 0;
        for (int i:
                integers) {
            intArray[currentIndex] = i;
            currentIndex++;
            if (currentIndex == 785) {
                currentIndex = 0;
                data.add(intArray);
            }
        }
    }

    public ArrayList<int[]> getData() {
        return this.data;
    }
}