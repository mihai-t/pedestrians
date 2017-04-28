package ro.ubbcluj.cs.io;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * @author Mihai Teletin
 */
public class TrainingDataReader {

    private static final Logger log = LoggerFactory.getLogger(TrainingDataReader.class);

    private static final String[] ALLOWED_EXTENSIONS = NativeImageLoader.ALLOWED_FORMATS;

    private static final File TRAINING_PATH = new File(System.getProperty("user.dir"), "../dataset/Training");

    private static DataNormalization dataNormalization;

    private static final int HEIGHT = 96;
    private static final int WIDTH = 48;
    private static final int CHANNELS = 1;
    private static final int CLASSES = 2;


    public static DataSetIterator getTrainingData(final int batchSize, final int seed, final double weight) throws IOException {
        final RecordReader recordReader = getImageRecordReader(seed, weight, 0);
        final RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, CLASSES);
        dataNormalization = normalize(iterator);

        return iterator;
    }

    public static DataSetIterator getValidationData(final int seed, final double weight) throws IOException {
        final RecordReader recordReader = getImageRecordReader(seed, 1 - weight, 1);
        final RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 1, 1, CLASSES);
        normalize(iterator);
        return iterator;
    }

    private static DataNormalization normalize(final DataSetIterator iterator) {
        final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);
        return scaler;
    }

    private static ImageRecordReader getImageRecordReader(final int seed, final double weight, final int index) throws IOException {
        final FileSplit filesInDir = new FileSplit(TRAINING_PATH, ALLOWED_EXTENSIONS, new Random(seed));
        log.info("Total number of files in dataset {}", filesInDir.length());
        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        final BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(seed), ALLOWED_EXTENSIONS, labelMaker);

        final InputSplit sample = filesInDir.sample(pathFilter, weight, 1 - weight)[index];

        log.info("Extracted {} samples", sample.length());

        final ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        recordReader.initialize(sample);

        return recordReader;
    }


    public static void main(String[] args) throws Exception {
        final DataSetIterator dataSetIterator = getTrainingData(200, 12345, 0.8);
        if (dataSetIterator.hasNext()) {
            final DataSet ds = dataSetIterator.next();
            System.out.println(ds);
        }

        final DataSetIterator dataSetIteratorTest = getValidationData(12345, 0.2);
        if (dataSetIterator.hasNext()) {
            final DataSet ds = dataSetIteratorTest.next();
            System.out.println(ds);
        }

    }

    public static DataNormalization getDataNormalization() {
        return dataNormalization;
    }
}

