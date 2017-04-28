package ro.ubbcluj.cs.test;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import ro.ubbcluj.cs.io.TrainingDataReader;

import java.io.*;

/**
 * @author Mihai.Teletin
 */
public class Model {
    private static final Log log = LogFactory.getLog(Model.class);
    private static final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader();
    private static final String NAME = "models/fourth architecture/fourth convolutional network50.zip";
    private static final String NORMALIZER_NAME = "models/normalizer/norm";
    private final MultiLayerNetwork multiLayerNetwork;
    private DataNormalization dataNormalization;

    public Model() throws IOException {
        multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(NAME);
        log.info("Loaded model");
        loadNormaliser();
        log.info("Loaded normaliser");
    }

    private void loadNormaliser() throws IOException {
        final File file = new File(NORMALIZER_NAME);
        if (file.exists()) {
            try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(NORMALIZER_NAME))) {
                try {
                    dataNormalization = (DataNormalization) in.readObject();
                } catch (ClassNotFoundException e) {
                    throw new RuntimeException(e.getMessage(), e);
                }
            }
        } else {
            TrainingDataReader.getTrainingData(1, 12345, 0.8);
            dataNormalization = TrainingDataReader.getDataNormalization();
            try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(NORMALIZER_NAME))) {
                out.writeObject(dataNormalization);
            }
        }
    }

    public double isPedestrian(final byte[] crop) throws IOException {
        final INDArray indArray = NATIVE_IMAGE_LOADER.asMatrix(new ByteArrayInputStream(crop));
        dataNormalization.transform(indArray);
        final INDArray output = multiLayerNetwork.output(indArray, false);
        // final double negative = output.getDouble(0);
        return output.getDouble(1);
    }

}
