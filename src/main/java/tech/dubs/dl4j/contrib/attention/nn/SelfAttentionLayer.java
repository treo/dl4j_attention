package tech.dubs.dl4j.contrib.attention.nn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/**
 * Self Attention Layer Implementation
 *
 * The implementation of mmul across time isn't the most efficient thing possible in nd4j, since the reshapes require
 * a copy, but it is the easiest to follow for now.
 *
 * TODO:
 *  - Optionally keep attention weights around for inspection
 *  - Handle Masking
 *
 * @author Paul Dubs
 */
public class SelfAttentionLayer extends BaseLayer<tech.dubs.dl4j.contrib.attention.conf.SelfAttentionLayer> {
    private IActivation softmax = new ActivationSoftmax();

    public SelfAttentionLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        Preconditions.checkState(input.rank() == 3,
            "3D input expected to RNN layer expected, got " + input.rank());

        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, training, workspaceMgr);

        long examples = input.size(0);
        long tsLength = input.size(2);
        long nOut = layerConf().getNOut();
        long nIn = layerConf().getNIn();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nIn, nOut}, 'f');

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        INDArray attentionWeights = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{examples, nOut, tsLength}, 'f');
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(attentionWeights, b, attentionWeights, 1));

        final long tads = input.tensorssAlongDimension(1, 2);
        for (int tad = 0; tad < tads; tad++) {
            final INDArray in = input.tensorAlongDimension(tad, 1, 2);
            final INDArray attentionWeight = attentionWeights.tensorAlongDimension(tad, 1, 2);
            final INDArray currentOutput = activations.tensorAlongDimension(tad, 1, 2);

            attentionWeight.addi(Nd4j.gemm(W, in, true, false));

            a.getActivation(attentionWeight, training);
            softmax.getActivation(attentionWeight, training);

            in.mmuli(attentionWeight.transposei(), currentOutput);
        }

        return activations;
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if(epsilon.ordering() != 'f' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('f');

        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, true, workspaceMgr);

        INDArray Wg = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bg = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        gradientsFlattened.assign(0);

        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.shape(), 'f');

        applyDropOutIfNecessary(true, workspaceMgr);

        long tsLength = input.size(2);
        long nOut = layerConf().getNOut();
        IActivation a = layerConf().getActivationFn();

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');


        final long tads = input.tensorssAlongDimension(1, 2);
        for (int tad = 0; tad < tads; tad++) {
            final INDArray attW = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{nOut, tsLength}, 'f');
            final INDArray attWPreA = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{nOut, tsLength}, 'f');;
            final INDArray attWPreS =  workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{nOut, tsLength}, 'f');


            final INDArray in = input.tensorAlongDimension(tad, 1, 2);
            final INDArray curEps = epsilon.tensorAlongDimension(tad, 1, 2);
            final INDArray curEpsOut = epsOut.tensorAlongDimension(tad, 1, 2);


            // Forward Pass with Caching of in-between results
            Nd4j.getExecutioner().exec(new BroadcastCopyOp(attW, b, attW, 0));

            attW.addi(Nd4j.gemm(W, in, true, false));
            attWPreA.assign(attW); // z: Pre Tanh
            attWPreS.assign(a.getActivation(attW, true)); // a: Pre Softmax
            softmax.getActivation(attW, true);

            final INDArray dLdy = Nd4j.gemm(curEps, in, true, false); // 	∂L/∂γ
            final INDArray dLda = softmax.backprop(attWPreS, dLdy).getFirst();// ∂L/∂a
            final INDArray dLdz = a.backprop(attWPreA, dLda).getFirst(); // ∂L/∂z
            //System.out.println(dLdz);

            // ∂L/∂b
            bg.addi(dLdz.sum(1).transposei());

            // ∂L/∂W
            Nd4j.gemm(in, dLdz, Wg, false, true, 1.0, 1.0);

            // ∂L/∂x: Part 1 - from multiplying with attention weight
            curEps.mmuli(attW, curEpsOut);
            // ∂L/∂x: Part 2 - from being used in attention weight calculation
            curEpsOut.addi(W.mmul(dLdz));
        }


        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, Wg);
        g.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, bg);

        epsOut = backpropDropOutIfPresent(epsOut);
        return new Pair<>(g, epsOut);
    }
}
