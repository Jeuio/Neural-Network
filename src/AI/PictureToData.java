package AI;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * This class is used to convert pictures into numeric data
 */
public class PictureToData {

    /**
     * Extract numeric data from an image
     * @param image the image
     * @return the data
     */
    public static double[] pictureToData(BufferedImage image) {
        double[] data = new double[image.getWidth() * image.getHeight()];
        //Iterates through all pixels in the image
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                Color color = new Color(image.getRGB(i, j));
                data[i * j] = (float)1 / 765 * color.getRed() + color.getGreen() + color.getBlue(); //Will return a value between 0 and 1 depending on the brightness of the pixel
            }
        }
        return data;
    }

    /**
     * Extract numeric data from a file leading to an image
     * @param file the file leading to an image
     * @return the data
     */
    public static float[] pictureToData(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        float[] data = new float[image.getWidth() * image.getHeight()];
        //Iterates through all pixels in the image
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                Color color = new Color(image.getRGB(i, j));
                data[i * j] = (float)1 / 765 * color.getRed() + color.getGreen() + color.getBlue(); //Will return a value between 0 and 1 depending on the brightness of the pixel
            }
        }
        return data;
    }

    /**
     * Extract numeric data from a path of the file leading to an image
     * @param filepath the path of the file leading to an image
     * @return the data
     */
    public static float[] pictureToData(String filepath) throws IOException {
        BufferedImage image = ImageIO.read(new File(filepath));
        float[] data = new float[image.getWidth() * image.getHeight()];
        //Iterates through all pixels in the image
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                Color color = new Color(image.getRGB(i, j));
                data[i * j] = (float)1 / 765 * color.getRed() + color.getGreen() + color.getBlue(); //Will return a value between 0 and 1 depending on the brightness of the pixel
            }
        }
        return data;
    }
}
