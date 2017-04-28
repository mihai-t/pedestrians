package ro.ubbcluj.cs.nn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * CONV(5x5, 25) -> NORM -> SUB(2x2, MAX) -> CONV(3x3, 50) -> SUB(2x2, MAX) -> NORM -> FC(400)
 *
 * @author Mihai Teletin
 */
public class ConvolutionalFour implements Network {


    public MultiLayerNetwork setupModel() {
        int layer = 0;

        final MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(1)
                .regularization(true).l2(0.0005)
                .learningRate(LEARNING_RATE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .useDropConnect(true)
                .leakyreluAlpha(0.02)
                .list()
                .layer(layer++, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .padding(2, 2)
                        .nOut(25)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(layer++, new LocalResponseNormalization.Builder().build())
                .layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(layer++, new ConvolutionLayer.Builder(3, 3)
                        .padding(1, 1)
                        .nOut(50)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(layer++, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(layer++, new LocalResponseNormalization.Builder().build())
                .layer(layer++, new DenseLayer.Builder().activation(Activation.RELU)
                        .name("dense")
                        .weightInit(WeightInit.RELU)
                        .nOut(400)
                        .dropOut(0.5)
                        .build())
                .layer(layer++, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backprop(true)
                .pretrain(false)
                .setInputType(InputType.convolutional(96, 48, 1));
        final MultiLayerConfiguration build = builder.build();
        final MultiLayerNetwork net = new MultiLayerNetwork(build);
        net.init();

        return net;
    }

    @Override
    public String getName() {
        return "fourth convolutional network";
    }
}
