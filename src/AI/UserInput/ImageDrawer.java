package AI.UserInput;

import AI.GUI.Draw;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageDrawer {

    private BufferedImage image;
    private File imageFile;

    public ImageDrawer(File imageFile) {
        try {
            this.image = ImageIO.read(imageFile);
            this.imageFile = imageFile;
            if (image.getWidth() != 28 || image.getHeight() != 28) {
                this.image = resizeImage(image, 28, 28);
            }
            for (int i = 0; i < image.getWidth(); i++) {
                for (int j = 0; j < image.getHeight(); j++) {
                    image.setRGB(i, j, Color.WHITE.getRGB());
                }
            }
            Draw.image = this.image;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public BufferedImage getImage() {
        return resizeImage(this.image, 28, 28);
    }

    public void draw(int x, int y) {
        try {
            image.setRGB(x, y, Color.BLACK.getRGB());
        } catch (ArrayIndexOutOfBoundsException e) {

        }
    }

    private BufferedImage resizeImage(BufferedImage image, int width, int height) {
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.drawImage(image, 0, 0, width, height, null);
        graphics2D.dispose();
        return resizedImage;
    }

}