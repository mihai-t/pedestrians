package ro.ubbcluj.cs.test;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Mihai Teletin on 28-Apr-17.
 */
public class ImageReader {


    private static final Log log = LogFactory.getLog(ImageReader.class);

    public static List<Crop> crop(final File image, final int offsetX, final int offsetY) throws IOException {

        final BufferedImage originalImage = ImageIO.read(image);

        final String fileName = image.getName().replace(".pgm", "");
        final int width = originalImage.getWidth();
        final int height = originalImage.getHeight();

        log.debug(String.format("Original image %s loaded. Dimension: %d x %d", fileName, width, height));
        final List<Crop> images = new ArrayList<>();
        for (int w = offsetX; w < width; w += 48) {
            for (int h = offsetY; h < height; h += 96) {
                if (w + 48 > width || h + 96 > height) {
                    continue;
                }
                final BufferedImage subImage = originalImage.getSubimage(w, h, 48, 96);

                final ByteArrayOutputStream output = new ByteArrayOutputStream();
                if (!ImageIO.write(subImage, "pnm", output)) {
                    log.error("failed to crop " + fileName);
                }

                images.add(new Crop(output.toByteArray(), w, h));
            }
        }

        return images;
    }

    static class Crop {
        byte[] content;
        int x, y;

        public Crop(byte[] content, int x, int y) {
            this.content = content;
            this.x = x;
            this.y = y;
        }
    }


}
