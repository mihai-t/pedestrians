package ro.ubbcluj.cs.test;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;

public class ImageTest {
    public static ImageFrame show(final byte[] image) throws IOException {
        final ImageFrame frame = new ImageFrame(image);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        return frame;
    }

    static class ImageFrame extends JFrame {
        ImageFrame(byte[] image) throws IOException {
            setTitle("Daimler");
            setSize(640, 480);
            add(new ImageComponent(image));
        }

        void showImage(final byte[] image) throws IOException {
            add(new ImageComponent(image));
            this.repaint();
        }

    }

    private static class ImageComponent extends JComponent {

        private Image image;

        ImageComponent(final byte[] src) throws IOException {
            image = ImageIO.read(new ByteArrayInputStream(src));
        }

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (image == null) {
                return;
            }
            g.drawImage(image, 0, 0, this);
            this.repaint();
        }

    }


}



