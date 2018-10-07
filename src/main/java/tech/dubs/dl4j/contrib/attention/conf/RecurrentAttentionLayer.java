package tech.dubs.dl4j.contrib.attention.conf;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import tech.dubs.dl4j.contrib.attention.nn.params.RecurrentQueryAttentionParamInitializer;

import java.util.Collection;
import java.util.Map;

/**
 * TODO: Memory Report, Configurable Activation for Attention
 *
 * @author Paul Dubs
 */
public class RecurrentAttentionLayer extends BaseRecurrentLayer {
    // No-Op Constructor for Deserialization
    public RecurrentAttentionLayer() { }

    private RecurrentAttentionLayer(Builder builder) {
        super(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {

        tech.dubs.dl4j.contrib.attention.nn.RecurrentAttentionLayer layer = new tech.dubs.dl4j.contrib.attention.nn.RecurrentAttentionLayer(conf);
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
        return RecurrentQueryAttentionParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName) {
        if(initializer().isWeightParam(this, paramName)){
            return l1;
        }else if(initializer().isBiasParam(this, paramName)){
            return l1Bias;
        }

        throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
    }

    @Override
    public double getL2ByParam(String paramName) {
        if(initializer().isWeightParam(this, paramName)){
            return l2;
        }else if(initializer().isBiasParam(this, paramName)){
            return l2Bias;
        }

        throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
    }


    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
       return null;
    }


    public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public RecurrentAttentionLayer build() {
            return new RecurrentAttentionLayer(this);
        }

    }
}
