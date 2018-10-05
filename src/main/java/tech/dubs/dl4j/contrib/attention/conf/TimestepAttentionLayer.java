package tech.dubs.dl4j.contrib.attention.conf;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import tech.dubs.dl4j.contrib.attention.nn.params.QueryAttentionParamInitializer;

import java.util.Collection;
import java.util.Map;

/**
 *
 *
 * @author Paul Dubs
 */
public class TimestepAttentionLayer extends BaseRecurrentLayer {
    // No-Op Constructor for Deserialization
    public TimestepAttentionLayer() { }

    private TimestepAttentionLayer(Builder builder) {
        super(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {

        tech.dubs.dl4j.contrib.attention.nn.TimestepAttentionLayer layer = new tech.dubs.dl4j.contrib.attention.nn.TimestepAttentionLayer(conf);
        layer.setListeners(iterationListeners);             //Set the iteration listeners, if any
        layer.setIndex(layerIndex);                         //Integer index of the layer

        layer.setParamsViewArray(layerParamsView);

        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        layer.setParamTable(paramTable);
        layer.setConf(conf);
        return layer;
    }

    @Override
    public ParamInitializer initializer() {
        return QueryAttentionParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName) {
            case QueryAttentionParamInitializer.WEIGHT_KEY:
            case QueryAttentionParamInitializer.QUERY_WEIGHT_KEY:
                return l1;
            case QueryAttentionParamInitializer.BIAS_KEY:
                return l1Bias;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName) {
            case QueryAttentionParamInitializer.WEIGHT_KEY:
            case QueryAttentionParamInitializer.QUERY_WEIGHT_KEY:
                return l2;
            case QueryAttentionParamInitializer.BIAS_KEY:
                return l2Bias;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer index = " + layerIndex
                + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                + inputType);
        }

        return InputType.recurrent(nIn, nIn);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer name = \"" + getLayerName()
                + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;
        final long tsLength = itr.getTimeSeriesLength();

        InputType outputType = getOutputType(-1, inputType);

        long numParams = initializer().numParams(this);
        int updaterStateSize = (int)getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if(getIDropout() != null){
            //Assume we dup the input for dropout
            trainSizeVariable += inputType.arrayElementsPerExample();
        }
        trainSizeVariable += outputType.arrayElementsPerExample() * tsLength;
        trainSizeVariable += itr.getSize() * outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, TimestepAttentionLayer.class, inputType, outputType)
            .standardMemory(numParams, updaterStateSize)
            .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
            .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
            .build();
    }


    public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public TimestepAttentionLayer build() {
            return new TimestepAttentionLayer(this);
        }

    }
}
