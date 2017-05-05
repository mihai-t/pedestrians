package ro.ubbcluj.cs.nn;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

/**
 * @author Mihai Teletin
 */
public interface Network {

    //0.003
    double LEARNING_RATE = 0.05;
    int SEED = 12345;
    double DROP_OUT = 0.5;

    static ParallelWrapper getGPUWrapper(final MultiLayerNetwork model) {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
        CudaEnvironment.getInstance().getConfiguration()
                .allowMultiGPU(true)
                .allowCrossDeviceAccess(true)
                .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
                .setMaximumGridSize(512)
                .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
                .setMaximumDeviceCache(4L * 1024 * 1024 * 1024L)
                .setMaximumHostCache(4L * 1024 * 1024 * 1024L)
                .setMaximumBlockSize(512);

        return new ParallelWrapper.Builder(model)
                .prefetchBuffer(24)
                .workers(4)
                .averagingFrequency(3)
                .reportScoreAfterAveraging(true)
                .useMQ(true)
                .build();
    }

    MultiLayerNetwork setupModel();

    String getName();


}
