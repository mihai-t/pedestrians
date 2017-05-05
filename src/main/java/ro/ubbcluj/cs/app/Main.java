package ro.ubbcluj.cs.app;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ro.ubbcluj.cs.io.TrainingDataReader;
import ro.ubbcluj.cs.nn.ConvolutionalOne;
import ro.ubbcluj.cs.nn.Network;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.time.Duration;
import java.time.Instant;

/**
 * @author Mihai Teletin
 */
public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    private static final int BATCH_SIZE = 75;
    private static final int EPOCHS = 100;

    private static final Network network = new ConvolutionalOne();
    private static final int INITIAL_EPOCH = 1;

    public static void main(String[] args) throws Exception {
        final Instant startApplication = Instant.now();

        final MultiLayerNetwork multiLayerNetwork;
        if (INITIAL_EPOCH == 1) {
            multiLayerNetwork = network.setupModel();
            log.info("Initialised model");
        } else {
            //load the previously saved model
            multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("models/" + network.getName() + (INITIAL_EPOCH - 1) + ".zip");
            log.info("Loaded model");
        }
        multiLayerNetwork.setListeners(new ScoreIterationListener(5));

        final ParallelWrapper gpuWrapper = Network.getGPUWrapper(multiLayerNetwork);
        log.info("Initialised GPU Wrapper");

        log.info("Loading training set");
        final DataSetIterator dataSetIterator = TrainingDataReader.getTrainingData(BATCH_SIZE, Network.SEED, 0.8);
        log.info("Training set loaded successfully");

        log.info("Loading validation set");
        final DataSetIterator dataSetIteratorValidation = TrainingDataReader.getValidationData(Network.SEED, 0.2);
        log.info("Validation set loaded successfully");
        final NumberFormat formatter = new DecimalFormat("#0.0000");

        for (int i = INITIAL_EPOCH; i <= EPOCHS; ++i) {
            log.info("Epoch {}", i);
            Instant start = Instant.now();
            gpuWrapper.fit(dataSetIterator);
            log.info("*** Completed epoch {}, time: {} min***", i, Duration.between(start, Instant.now()).toMinutes());
            saveModel(multiLayerNetwork, network.getName() + i);
            log.info("Evaluate model....");
            start = Instant.now();
            final Evaluation eval = new Evaluation(dataSetIteratorValidation.getLabels());
            dataSetIteratorValidation.reset();
            while (dataSetIteratorValidation.hasNext()) {
                final DataSet ds = dataSetIteratorValidation.next();
                final INDArray output = multiLayerNetwork.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }

            final String stats = eval.stats();
            log.info(stats);


            try (FileWriter fw = new FileWriter("results/result_" + network.getName() + ".txt", true)) {
                fw.write("Epoch " + i);
                fw.write("\n");
                fw.write("\n");

                final String acc = formatter.format(eval.accuracy());
                final String recall = formatter.format(eval.recall(1));
                final String precision = formatter.format(eval.precision(1));
                final String f1 = formatter.format(eval.f1(1));

                fw.write("Accuracy:" + acc);
                fw.write("\n");
                fw.write("prec on " + eval.getClassLabel(1) + ":" + precision);
                fw.write("\n");
                fw.write("recall on " + eval.getClassLabel(1) + ":" + recall);
                fw.write("\n");
                fw.write("f on " + eval.getClassLabel(1) + ":" + f1);
                fw.write("\n");
                fw.write("========================================================================");
                fw.write("\n");
                fw.write(eval.confusionToString());
                fw.write("========================================================================");
                fw.write("\n");
            }

            log.info("*** Evaluation for epoch {} took {} min***", i, Duration.between(start, Instant.now()).toMinutes());
        }

        log.info("*** Completed {} epochs, batch size: {}, time: {} hours***", EPOCHS, BATCH_SIZE, Duration.between(startApplication, Instant.now()).toHours());


    }

    private static void saveModel(final MultiLayerNetwork model, final String fileName) throws IOException {
        final File locationToSave = new File("models/" + fileName + ".zip");
        ModelSerializer.writeModel(model, locationToSave, true);
    }

}
