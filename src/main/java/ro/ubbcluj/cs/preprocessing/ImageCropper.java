package ro.ubbcluj.cs.preprocessing;/* ro.ubbcluj.cs.preprocessing.ImageCropper.java */

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Used to crop all the negative samples in order to match
 * the resolution of the positive samples
 *
 * @author Mihai Teletin
 */
public class ImageCropper {

    private static final Log log = LogFactory.getLog(ImageCropper.class);

    private static void crop(final File image) throws IOException {


        final BufferedImage originalImage = ImageIO.read(image);

        final String fileName = image.getName().replace(".pgm", "");
        final int width = originalImage.getWidth();
        final int height = originalImage.getHeight();

        log.debug(String.format("Original image %s loaded. Dimension: %d x %d", fileName, width, height));

        for (int w = 0; w < width; w += 48) {
            for (int h = 0; h < height; h += 96) {
                if (w + 48 > width || h + 96 > height) {
                    continue;
                }
                final BufferedImage subImage = originalImage.getSubimage(w, h, 48, 96);
                log.debug(String.format("Cropped Image Dimension: %d x %d", subImage.getWidth(), subImage.getHeight()));

                final File outputFile = new File(System.getProperty("user.dir"), "../dataset/cropped/" + fileName + "_" + w + "_" + h + ".pgm");

                if (!ImageIO.write(subImage, "pnm", outputFile)) {
                    log.error("failed to crop " + fileName);
                }
            }
        }


    }

    public static void main(String[] args) throws IOException {
        final URI uri = new File(System.getProperty("user.dir"), "../dataset/original/NonPedestrian").toURI();

        Files.list(Paths.get(uri)).parallel().forEach(s -> {
            try {
                crop(new File(s.toString()));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
}