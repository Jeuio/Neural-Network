package AI.GUI;

import AI.UserInput.MouseListener;

import javax.swing.*;
import java.awt.*;

public class GUI {

    public GUI() {
        JFrame frame = new JFrame();
        frame.setSize(new Dimension(840, 840));
        JLabel draw = new Draw(frame.getWidth(), frame.getHeight());
        draw.setBounds(0, 0, frame.getWidth(), frame.getHeight());

        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(draw);
        frame.addMouseListener(new MouseListener());
        frame.addMouseMotionListener(new MouseListener());
        frame.setVisible(true);
    }
}
