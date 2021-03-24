package AI.UserInput;

import AI.GUI.Draw;

import javax.imageio.ImageIO;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;

public class MouseListener implements java.awt.event.MouseListener {

    private boolean pressed = false;
    public static int x = 0;
    public static int y = 0;
    public ImageDrawer imageDrawer;
    private Thread thread;

    {
        try {
            imageDrawer = new ImageDrawer(ImageIO.read(new File("src\\AI\\UserInput\\image.jpg")));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void mouseClicked(MouseEvent mouseEvent) {
        //System.out.println("x = " + x + " | " + Math.round(x / 30f));
        //System.out.println("y = " + y + " | " + Math.round(y / 30f));
    }

    @Override
    public void mousePressed(MouseEvent mouseEvent) {
        this.pressed = true;
        thread = new Thread(() -> {
            while(pressed) {
                x = Math.round(mouseEvent.getX() / 30f);
                y = Math.round(mouseEvent.getY() / 30f);
                System.out.println("x = " + x);
                System.out.println("y = " + y);
                imageDrawer.draw(x, y);
                Draw.image = imageDrawer.getImage();
            }
        });
        thread.start();
    }

    @Override
    public void mouseReleased(MouseEvent mouseEvent) {
        this.pressed = false;
        thread.stop();
    }

    @Override
    public void mouseEntered(MouseEvent mouseEvent) {

    }

    @Override
    public void mouseExited(MouseEvent mouseEvent) {

    }
}
