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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Mihai.Teletin
 */
public class Model {
    private static final Log log = LogFactory.getLog(Model.class);
    private static final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader();
    private static final List<String> MODEL_NAMES = Arrays.asList(
            "models/first architecture/first convolutional network85.zip",
            "models/second architecture/second convolutional network64.zip",
            "models/third architecture/third convolutional network58.zip",
            "models/fourth architecture/fourth convolutional network10.zip",
            "models/first architecture improve/first convolutional network10.zip"
    );

    private static final String NORMALIZER_NAME = "models/normalizer/norm";
    private final List<MultiLayerNetwork> multiLayerNetworks = new ArrayList<>();
    private DataNormalization dataNormalization;

    public Model() throws IOException {
        for (String name : MODEL_NAMES) {
            multiLayerNetworks.add(ModelSerializer.restoreMultiLayerNetwork(name));
        }
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

        double res = 0;

        for (MultiLayerNetwork multiLayerNetwork : multiLayerNetworks) {
            final INDArray output = multiLayerNetwork.output(indArray);
            res += output.getDouble(1);
        }

        // final double negative = output.getDouble(0);
        return res / multiLayerNetworks.size();
    }


}
