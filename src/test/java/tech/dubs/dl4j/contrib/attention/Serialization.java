package tech.dubs.dl4j.contrib.attention;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tech.dubs.dl4j.contrib.attention.conf.SelfAttentionLayer;

public class Serialization {
    @Test
    public void testSelfAttentionSerialization(){
        int nIn = 3;
        int nOut = 5;
        int layerSize = 8;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .updater(new NoOp())
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new LSTM.Builder().nOut(layerSize).build())
                .layer(new SelfAttentionLayer.Builder().build())
                .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .setInputType(InputType.recurrent(nIn))
                .build();

        final String json = conf.toJson();
        final String yaml = conf.toYaml();

        final MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(json);
        final MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(yaml);

        Assert.assertEquals(conf, fromJson);
        Assert.assertEquals(conf, fromYaml);
    }
}
