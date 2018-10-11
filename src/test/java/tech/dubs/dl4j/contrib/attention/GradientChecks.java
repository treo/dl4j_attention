package tech.dubs.dl4j.contrib.attention;

import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import tech.dubs.dl4j.contrib.attention.conf.RecurrentAttentionLayer;
import tech.dubs.dl4j.contrib.attention.conf.SelfAttentionLayer;
import tech.dubs.dl4j.contrib.attention.conf.TimestepAttentionLayer;
import tech.dubs.dl4j.contrib.attention.nn.AttentionMechanism;

import java.util.Random;

import static org.junit.Assert.assertTrue;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

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
    public void getViewAssign(){
        final INDArray reshape = Nd4j.linspace(1, 6, 6).reshape('c', 3, 2);

        final INDArray target = Nd4j.zeros(2, 6, 2);
        final INDArray targetView = target.get(point(1), all(), point(0));
        final INDArray targetView2 = target.tensorAlongDimension(0, 1);


        System.out.println(reshape);
        System.out.println(reshape.reshape(3*2, 1));
        targetView.assign(reshape.reshape(3*2, 1));
        System.out.println(targetView);
        targetView2.assign(reshape.reshape(3*2, 1));
        System.out.println(targetView2);

        System.out.println(target);

    }


    @Test
    public void repl(){
        final int attentionHeads = 2;
        final INDArray keys = Nd4j.linspace(1, 24, 24).reshape(2, 3, 4);
        final INDArray queries = Nd4j.linspace(1, 18, 18).reshape(2, 3, 3);

        final INDArray W = Nd4j.ones(3, attentionHeads);
        final INDArray Q = Nd4j.linspace(1, 3*attentionHeads, 3*attentionHeads).reshape(3, attentionHeads);
        final INDArray b = Nd4j.ones(1, attentionHeads);

        System.out.println(queries);
        System.out.println(Q);
        final AttentionMechanism mechanism = new AttentionMechanism(Q, W, b, new ActivationIdentity(), LayerWorkspaceMgr.noWorkspaces(), true);

        final INDArray query = mechanism.query(queries, keys, keys, null);
        System.out.println(mechanism.backprop(query, queries, keys, keys, null));
    }

    @Test
    public void testSelfAttentionLayer() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;
        int attentionHeads = 2;

        Random r = new Random(12345);
        for (int mb : new int[]{2, 3,1}) {
            for (boolean inputMask : new boolean[]{true, false}) {
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

                String name = "testSelfAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new SelfAttentionLayer.Builder().nOut(attentionHeads).build())
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE,in, labels, inMask, null, false, -1,
                        null //Sets.newHashSet(  /*"1_b", "1_W",* "1_WR", "1_WQR", "1_WQ", "1_bQ",*/ "2_b", "2_W" ,"0_W", "0_RW", "0_b"/**/)
                );
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
                        .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
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
    public void testRecurrentAttentionLayer() {
        int nIn = 3;
        int nOut = 5;
        int tsLength = 4;
        int layerSize = 8;
        int attentionHeads = 7;


        Random r = new Random(12345);
        for (int mb : new int[]{3, 2, 7}) {
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

                String name = "testRecurrentAttentionLayer() - mb=" + mb + ", tsLength = " + tsLength + ", maskType=" + maskType;
                System.out.println("Starting test: " + name);


                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .activation(Activation.TANH)
                        .updater(new NoOp())
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(new LSTM.Builder().nOut(layerSize).build())
                        .layer(new LastTimeStep(new RecurrentAttentionLayer.Builder().nOut(7).build()))
                        .layer(new OutputLayer.Builder().nOut(nOut).activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .setInputType(InputType.recurrent(nIn))
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                //System.out.println("Original");
                boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                        DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in, labels, inMask, null, false, -1, null
                        //Sets.newHashSet(  /*"1_b", "1_W",* "1_WR", "1_WQR", "1_WQ", "1_bQ",*/ "2_b", "2_W" ,"0_W", "0_RW", "0_b"/**/)
                );
                assertTrue(name, gradOK);

                /*System.out.println("in: " + in.shapeInfoToString());
                System.out.println("Einzeln");
                final long l = in.tensorssAlongDimension(1,2);
                for (int i = 0; i < l; i++) {
                    gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, in.tensorAlongDimension(i, 1, 2).reshape(1, nIn, tsLength), labels.tensorAlongDimension(i, 1).reshape(1, nOut), inMask, null, false, -1,
                            Sets.newHashSet(  "1_b", *//*"1_W",*//* "1_WR", "1_WQR", "1_WQ", "1_bQ", "3_b", "3_W", "0_W", "0_RW", "0_b"));
                    assertTrue(name, gradOK);
                }


                assertTrue(name, gradOK);*/
            }
        }
    }
}