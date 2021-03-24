package AI.UserInput;

import AI.GUI.Draw;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageDrawer {

    private final BufferedImage image;

    public ImageDrawer(BufferedImage image) {
        if (image.getWidth() != 28 || image.getHeight() != 28) {
            this.image = resizeImage(image, 28, 28);
        } else {
            this.image = image;
        }
        Draw.image = this.image;
    }

    public BufferedImage getImage() {
        return image;
    }

    public void draw(int x, int y) {
        image.setRGB(x, y, Color.BLACK.getRGB());
    }

    private BufferedImage resizeImage(BufferedImage image, int width, int height) {
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.drawImage(image, 0, 0, width, height, null);
        graphics2D.dispose();
        return resizedImage;
    }

}