package ro.ubbcluj.cs.test;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.List;

public class ImageTest {
    public static ImageFrame show(final byte[] image, List<ImageReader.Crop> positives) throws IOException {
        final ImageFrame frame = new ImageFrame(image, positives);
    //    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        return frame;
    }

    static class ImageFrame extends JFrame {
        ImageFrame(byte[] image, List<ImageReader.Crop> positives) throws IOException {
            setTitle("Daimler");
            setSize(640, 480);
            add(new ImageComponent(image, positives));
        }

//        void showImage(final byte[] image) throws IOException {
//            add(new ImageComponent(image));
//            this.repaint();
//        }

    }

    private static class ImageComponent extends JComponent {

        private Image image;
        private List<ImageReader.Crop> positives;

        ImageComponent(final byte[] src, List<ImageReader.Crop> positives) throws IOException {
            image = ImageIO.read(new ByteArrayInputStream(src));
            this.positives = positives;
        }

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (image == null) {
                return;
            }
            g.drawImage(image, 0, 0, this);

            final Graphics2D g2d = (Graphics2D) g;
            g2d.setColor(Color.RED);
            g2d.setStroke(new BasicStroke(2f));


            for (ImageReader.Crop crop : positives) {
                g2d.drawRect(crop.x, crop.y, 48, 96);
            }
            this.repaint();
        }

    }


}



