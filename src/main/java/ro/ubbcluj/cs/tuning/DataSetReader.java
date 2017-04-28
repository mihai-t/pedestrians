package ro.ubbcluj.cs.tuning;

import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ro.ubbcluj.cs.nn.Network;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * @author Mihai Teletin
 */
public class DataSetReader {
    private static final Logger log = LoggerFactory.getLogger(DataSetReader.class);

    private static final File TRAINING_PATH = new File(System.getProperty("user.dir"), "../dataset/tuning");
    private static final BaseImageLoader loader = new NativeImageLoader(96, 48, 1);
    private DataNormalization normalization;

    public DataSetIterator loadTestingSet() throws IOException {
        final List<Pair<INDArray, INDArray>> set = loadSamples("Testing/Pedestrian", -1, Nd4j.create(new double[]{0, 1}), normalization);
        log.info("Loaded {} positives", set.size());
        set.addAll(loadSamples("Testing/NonPedestrian", -1, Nd4j.create(new double[]{1, 0}), normalization));
        log.info("Loaded {} samples", set.size());
        return new INDArrayDataSetIterator(set, 1);
    }

    public DataSetIterator loadTrainingSet(int batchSize) throws IOException {
        final List<Pair<INDArray, INDArray>> set = loadSamples("Pedestrian", -1, Nd4j.create(new double[]{0, 1}), null);
        log.info("Loaded {} positives", set.size());
        set.addAll(loadSamples("NonPedestrian", set.size(), Nd4j.create(new double[]{1, 0}), null));
        log.info("Loaded {} samples", set.size());
        Collections.shuffle(set);
        final INDArrayDataSetIterator indArrayDataSetIterator = new INDArrayDataSetIterator(set, batchSize);
        normalization = normalize(indArrayDataSetIterator);
        return indArrayDataSetIterator;
    }

    private static DataNormalization normalize(final DataSetIterator iterator) {
        final DataNormalization scaler = new ImagePreProcessingScaler(-1, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);
        return scaler;
    }

    private List<Pair<INDArray, INDArray>> loadSamples(String path, int limit, INDArray label, DataNormalization normalization) throws IOException {
        String target = TRAINING_PATH + "/" + path;
        final List<Pair<INDArray, INDArray>> list = new ArrayList<>();
        final Random random = new Random(Network.SEED);
        final List<File> files = Files.walk(Paths.get(target))
                .parallel()
                .filter(s -> random.nextDouble() > 0.2)//make it random
                .limit(limit == -1 ? Long.MAX_VALUE : limit)
                .map(s -> new File(s.toUri()))
                .collect(Collectors.toList());
        for (final File file : files) {
            final INDArray array = loader.asMatrix(file);
            if (normalization != null) {
                normalization.transform(array);
            }
            list.add(new Pair<>(array, label));
        }
        return list;
    }
}
