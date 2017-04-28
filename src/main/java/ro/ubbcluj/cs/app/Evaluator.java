package ro.ubbcluj.cs.app;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ro.ubbcluj.cs.io.TrainingDataReader;
import ro.ubbcluj.cs.nn.ConvolutionalOne;
import ro.ubbcluj.cs.nn.Network;

import java.io.FileWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.time.Duration;
import java.time.Instant;

/**
 * @author Mihai Teletin
 */
public class Evaluator {

    private static final Logger log = LoggerFactory.getLogger(Evaluator.class);


    private static final int EPOCHS = 50;

    private static final Network network = new ConvolutionalOne();

    public static void main(String[] args) throws Exception {

        log.info("Loading validation set");
        final DataSetIterator dataSetIteratorValidation = TrainingDataReader.getValidationData(Network.SEED, 0.2);
        log.info("Validation set loaded successfully");
        log.info("Labels: " + dataSetIteratorValidation.getLabels());

        final NumberFormat formatter = new DecimalFormat("#0.0000");
        for (int i = 1; i <= EPOCHS; ++i) {
            log.info("Evaluation for epoch {}", i);
            final MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("models/" + network.getName() + i + ".zip");
            log.info("Loaded model {}", network.getName() + i);
            Instant start = Instant.now();
            log.info("Evaluate model....");
            final Evaluation eval = new Evaluation(dataSetIteratorValidation.getLabels());
            dataSetIteratorValidation.reset();
            while (dataSetIteratorValidation.hasNext()) {
                final DataSet ds = dataSetIteratorValidation.next();
                final INDArray output = multiLayerNetwork.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            final String stats = eval.stats();
            log.info(stats);

            try (FileWriter fw = new FileWriter("results/reeval_result_" + network.getName() + ".txt", true)) {
                fw.write("Epoch " + i);
                fw.write("\n");
                fw.write("\n");
                fw.write(stats);
                fw.write("\n");

                fw.write("prec on " + eval.getClassLabel(0) + ":" + formatter.format(eval.precision(0)));
                fw.write("\n");
                fw.write("prec on " + eval.getClassLabel(1) + ":" + formatter.format(eval.precision(1)));
                fw.write("\n");
                fw.write("recall on " + eval.getClassLabel(0) + ":" + formatter.format(eval.recall(0)));
                fw.write("\n");
                fw.write("recall on " + eval.getClassLabel(1) + ":" + formatter.format(eval.recall(1)));
                fw.write("\n");
                fw.write("f on " + eval.getClassLabel(0) + ":" + formatter.format(eval.f1(0)));
                fw.write("\n");
                fw.write("f on " + eval.getClassLabel(1) + ":" + formatter.format(eval.f1(1)));
                fw.write("\n");
                fw.write(eval.confusionToString());
                fw.write("\n");
                fw.write("========================================================================");
                fw.write("\n");
                fw.write("\n");
            }

            log.info("*** Evaluation for epoch {} took {} min***", i, Duration.between(start, Instant.now()).toMinutes());
        }


    }

}
