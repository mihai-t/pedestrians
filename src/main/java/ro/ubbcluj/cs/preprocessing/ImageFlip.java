package ro.ubbcluj.cs.preprocessing;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.imageio.ImageIO;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Used to flip horizontally all the positive samples in order to double
 * the size of the positive samples training set
 *
 * @author Mihai Teletin
 */
public class ImageFlip {

    private static final Log log = LogFactory.getLog(ImageCropper.class);

    private static void flipHorizontal(final File image) throws IOException {
        BufferedImage pgmImage = ImageIO.read(image);
        final String fileName = image.getName().replace(".pgm", "");
        //  log.debug(String.format("Original image %s loaded.", fileName));

        final File outputFile = new File(System.getProperty("user.dir"), "../dataset/flipped/" + fileName + "_flipped.pgm");

        // Flip the image vertically
        final AffineTransform tx = AffineTransform.getScaleInstance(-1, 1);
        tx.translate(-pgmImage.getWidth(null), 0);
        final AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
        pgmImage = op.filter(pgmImage, null);

        if (!ImageIO.write(pgmImage, "pnm", outputFile)) {
            log.error("failed to flip " + fileName);
        }

    }

    public static void main(String[] args) throws IOException {
        final URI uri = new File(System.getProperty("user.dir"), "../dataset/original/Pedestrian").toURI();

        Files.list(Paths.get(uri)).parallel().forEach(s -> {
            try {
                flipHorizontal(new File(s.toString()));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
}
