package AI.GUI;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Draw extends JLabel {

    public static BufferedImage image;
    public static String guess = "not guessed yet";
    private final int width;
    private final int height;


    public Draw(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);

        g2d.drawImage(image, 0, 0, width, height, null);
        g2d.drawString(guess, 10, 10);

        repaint();
    }
}
