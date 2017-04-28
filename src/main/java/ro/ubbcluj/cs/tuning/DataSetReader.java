package ro.ubbcluj.cs.tuning;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * @author Mihai Teletin
 */
public class DataSetReader {

    private static final String[] ALLOWED_EXTENSIONS = NativeImageLoader.ALLOWED_FORMATS;

    private static final File TRAINING_PATH = new File(System.getProperty("user.dir"), "../dataset/tuning/Training");
    private static final File VALIDATION_PATH = new File(System.getProperty("user.dir"), "../dataset/tuning/Testing");
    private static final int HEIGHT = 96;
    private static final int WIDTH = 48;
    private static final int CHANNELS = 1;
    private static final int CLASSES = 2;
    private static DataNormalization dataNormalization;

    private static DataNormalization normalize(final DataSetIterator iterator) {
        final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);
        return scaler;
    }

    private static ImageRecordReader getImageRecordReader(final File path, boolean balance) throws IOException {
        final FileSplit filesInDir = new FileSplit(path, ALLOWED_EXTENSIONS, new Random());
        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        final ImageRecordReader recordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        if (balance) {
            final BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(), ALLOWED_EXTENSIONS, labelMaker);

            final InputSplit sample = filesInDir.sample(pathFilter, 1)[0];
            recordReader.initialize(sample);
        } else {
            recordReader.initialize(filesInDir);
        }

        return recordReader;
    }

    public static DataNormalization getDataNormalization() {
        return dataNormalization;
    }

    public DataSetIterator loadTrainingSet(final int batchSize) throws IOException {
        final RecordReader recordReader = getImageRecordReader(TRAINING_PATH, true);
        final RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, CLASSES);
        dataNormalization = normalize(iterator);

        return iterator;
    }

    public DataSetIterator loadTestingSet() throws IOException {
        final RecordReader recordReader = getImageRecordReader(VALIDATION_PATH, false);
        final RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 1, 1, CLASSES);
        normalize(iterator);
        return iterator;
    }
}

