package AI.UserInput;

import AI.GUI.Draw;

import javax.imageio.ImageIO;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;

public class MouseListener extends MouseAdapter {

    public static int x = 0;
    public static int y = 0;
    public static ImageDrawer imageDrawer;

    {
        imageDrawer = new ImageDrawer(new File("src\\AI\\UserInput\\image.jpg"));
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        x = (int) Math.floor((e.getX() - 7) / 30f);
        y = (int) Math.floor((e.getY() - 30) / 30f);
        imageDrawer.draw(x, y);
        Draw.image = imageDrawer.getImage();
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        x = (int) Math.floor((e.getX() - 7) / 30f);
        y = (int) Math.floor((e.getY() - 30) / 30f);
        imageDrawer.draw(x, y);
        Draw.image = imageDrawer.getImage();
    }
}