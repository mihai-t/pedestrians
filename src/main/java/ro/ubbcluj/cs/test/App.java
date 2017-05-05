package ro.ubbcluj.cs.test;

import org.apache.commons.compress.utils.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Mihai.Teletin
 */
public class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    private static final List<String> testImages = Arrays.asList("00m_25s_971567u.pgm", "21m_23s_798629u.pgm", "05m_38s_172861u.pgm");

    public static void main(String[] args) throws Exception {

        final URI uri = new File(System.getProperty("user.dir"), "../dataset/Test").toURI();

        final List<File> list = Files.list(Paths.get(uri))
              //  .filter(s -> testImages.contains(s.getFileName().toString()))
                .map(s -> new File(s.toString()))
                .collect(Collectors.toList());

        Collections.shuffle(list);
        final Model model = new Model();

        for (final File file : list) {
            final List<ImageReader.Crop> positives = new ArrayList<>();
            log.info(file.getName());

            final int strideX = 15, strideY = 20;

            final List<ImageReader.Crop> crops = ImageReader.crop(file, strideX, strideY);

            log.info("loaded {} crops", crops.size());
            for (ImageReader.Crop crop : crops) {
                final double pedestrian = model.isPedestrian(crop.content);
                if (pedestrian > 0.99) {
                    positives.add(crop);
                }
            }

            ImageTest.show(IOUtils.toByteArray(new FileInputStream(file)), positives);
//            Thread.sleep(5000);
//
//
//            if (show != null) {
//                show.setVisible(false);
//                show.dispose();
//            }
        }

    }
}
