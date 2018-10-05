package tech.dubs.dl4j.contrib.attention;

import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tech.dubs.dl4j.contrib.attention.conf.SelfAttentionLayer;
import tech.dubs.dl4j.contrib.attention.conf.TimestepAttentionLayer;

import java.util.Random;

import static org.junit.Assert.assertTrue;

public class GradientChecks {
    private static final boolean PRINT_RESULTS = false;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }


    @Test
    public void repl(){
        final INDArray a = Nd4j.linspace(1, 3, 3).reshape(1, 3);
        final INDArray b = Nd4j.linspace(1, 3, 3).reshape(3, 1);
        final INDArray weights = a.broadcast(3, 3).addiColumnVector(b);

        final INDArray input = Nd4j.linspace(1, 9, 9).reshape(3, 3);


        System.out.println(input);
        System.out.println(weights);
        System.out.println(input.mmul(weights.transposei()));
    }

    @Test
    public void testSelfAttentionLayer() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;
        int attentionHeads = 7;

        Random r = new Random(12345);
        for (int mb : new int[]{7, 3, 1}) {
            for (boolean inputMask : new boolean[]{false, true}) {
                INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                INDArray labels = Nd4j.create(mb, nOut);
                for (int i = 0; i < mb; i++) {
                    labels.putScalar(i, r.nextInt(nOut), 1.0);
                }
                String maskType = (inputMask ? "inputMask" : "none");

                INDArray inMask = null;
                if (inputMask) {
                    inMask = Nd4j.ones(mb, attentionHeads);
                    for (int i = 0; i < mb; i++) {
                        int firstMaskedStep = attentionHeads - 1 - i;
                        if (firstMaskedStep == 0) {
                            firstMaskedStep = attentionHeads;
                        }
                        for (int j = firstMaskedStep; j < attentionHeads; j++) {
                            inMask.putScalar(i, j, 0.0);
                        }
                    }
                }

                String name = "testSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new SelfAttentionLayer.Builder().nOut(attentionHeads).build())
                        .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                assertTrue(name, gradOK);
            }
        }
    }

    @Test
    public void testTimestepAttentionLayer() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;
        int attentionHeads = 7;

        Random r = new Random(12345);
        for (int mb : new int[]{1, 3, 7}) {
            for (boolean inputMask : new boolean[]{false}) {
                INDArray in = Nd4j.rand(new int[]{mb, nIn, tsLength});
                INDArray labels = Nd4j.create(mb, nOut);
                for (int i = 0; i < mb; i++) {
                    labels.putScalar(i, r.nextInt(nOut), 1.0);
                }
                String maskType = (inputMask ? "inputMask" : "none");

                INDArray inMask = null;
                if (inputMask) {
                    inMask = Nd4j.ones(mb, tsLength);
                    for (int i = 0; i < mb; i++) {
                        int firstMaskedStep = tsLength - 1 - i;
                        if (firstMaskedStep == 0) {
                            firstMaskedStep = tsLength;
                        }
                        for (int j = firstMaskedStep; j < tsLength; j++) {
                            inMask.putScalar(i, j, 0.0);
                        }
                    }
                }

                String name = "testTimestepAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new TimestepAttentionLayer.Builder().nOut(1).build())
                        .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null);
                assertTrue(name, gradOK);
            }
        }
    }
}