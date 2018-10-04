package tech.dubs.dl4j.contrib.attention;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 *  nOut = Number of Attention Heads
 *
 *  The self attention layer is the simplest of attention layers. It takes a recurrent input, and returns a fixed size
 *  dense output. For this it requires just the same parameters as a dense layer, since for the most learning part of it
 *  it actually is a dense layer.
 *
 * @author Paul Dubs
 */
public class SelfAttentionLayer extends FeedForwardLayer {
    // No-Op Constructor for Deserialization
    public SelfAttentionLayer() { }

    private SelfAttentionLayer(Builder builder) {
        super(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {

        SelfAttentionLayerImpl layer = new SelfAttentionLayerImpl(conf);
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
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer index = " + layerIndex
                + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                + inputType);
        }

        return InputType.recurrent(nIn, nOut);
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

        return new LayerMemoryReport.Builder(layerName, SelfAttentionLayer.class, inputType, outputType)
            .standardMemory(numParams, updaterStateSize)
            .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
            .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
            .build();
    }


    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public SelfAttentionLayer build() {
            return new SelfAttentionLayer(this);
        }
    }
}
