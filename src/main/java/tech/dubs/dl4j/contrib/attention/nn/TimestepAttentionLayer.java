package tech.dubs.dl4j.contrib.attention.nn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
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
import tech.dubs.dl4j.contrib.attention.nn.params.QueryAttentionParamInitializer;

/**
 * Timestep Attention Layer Implementation
 *
 *
 * TODO:
 *  - Optionally keep attention weights around for inspection
 *  - Handle Masking
 *
 * @author Paul Dubs
 */
public class TimestepAttentionLayer extends BaseLayer<tech.dubs.dl4j.contrib.attention.conf.TimestepAttentionLayer> {
    private IActivation softmax = new ActivationSoftmax();

    public TimestepAttentionLayer(NeuralNetConfiguration conf) {
        super(conf);
        final long nOut = layerConf().getNOut();
        if(nOut > 1){
            throw new IllegalArgumentException("TimestepAttentionLayer currently doesn't support more than one Attention head! Got: "+ nOut);
        }
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

        INDArray W = getParamWithNoise(QueryAttentionParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray Wq = getParamWithNoise(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(QueryAttentionParamInitializer.BIAS_KEY, training, workspaceMgr);

        long examples = input.size(0);
        long tsLength = input.size(2);
        long nIn = layerConf().getNIn();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nIn, tsLength}, 'f');

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        final long tads = input.tensorssAlongDimension(1, 2);
        for (int tad = 0; tad < tads; tad++) {
            INDArray perTsAttW = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{tsLength, 1}, 'f');
            INDArray allTsAttW = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{ 1, tsLength}, 'f');
            INDArray attW = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{ tsLength, tsLength}, 'f');

            Nd4j.getExecutioner().exec(new BroadcastCopyOp(allTsAttW, b, allTsAttW, 0));

            final INDArray in = input.tensorAlongDimension(tad, 1, 2);
            final INDArray currentOutput = activations.tensorAlongDimension(tad, 1, 2);

            allTsAttW.addi(Nd4j.gemm(W, in, true, false));
            perTsAttW.assign(Nd4j.gemm(in, Wq, true, false));

            allTsAttW.broadcast(attW).addiColumnVector(perTsAttW);

            a.getActivation(attW, training);
            softmax.getActivation(attW, training);

            currentOutput.assign(in.mmul(attW.transposei()));
        }

        return activations;
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if(epsilon.ordering() != 'f' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('f');

        INDArray W = getParamWithNoise(QueryAttentionParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray Wq = getParamWithNoise(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY, true, workspaceMgr);
        INDArray b = getParamWithNoise(QueryAttentionParamInitializer.BIAS_KEY, true, workspaceMgr);

        INDArray Wg = gradientViews.get(QueryAttentionParamInitializer.WEIGHT_KEY);
        INDArray Wqg = gradientViews.get(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY);
        INDArray bg = gradientViews.get(QueryAttentionParamInitializer.BIAS_KEY);
        gradientsFlattened.assign(0);

        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.shape(), 'f');

        applyDropOutIfNecessary(true, workspaceMgr);

        long tsLength = input.size(2);
        IActivation a = layerConf().getActivationFn();

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');


        final long tads = input.tensorssAlongDimension(1, 2);
        for (int tad = 0; tad < tads; tad++) {
            INDArray perTsAttW = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{tsLength, 1}, 'f');
            INDArray allTsAttW = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{ 1, tsLength}, 'f');
            INDArray attW = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{ tsLength, tsLength}, 'f');
            INDArray attWPreA = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{tsLength, tsLength}, 'f');;
            INDArray attWPreS =  workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{tsLength, tsLength}, 'f');

            INDArray in = input.tensorAlongDimension(tad, 1, 2);
            INDArray curEps = epsilon.tensorAlongDimension(tad, 1, 2);
            INDArray curEpsOut = epsOut.tensorAlongDimension(tad, 1, 2);

            // Forward Pass with Caching of in-between results
            Nd4j.getExecutioner().exec(new BroadcastCopyOp(allTsAttW, b, allTsAttW, 0));

            allTsAttW.addi(Nd4j.gemm(W, in, true, false));
            perTsAttW.assign(Nd4j.gemm(in, Wq, true, false));
            allTsAttW.broadcast(attW).addiColumnVector(perTsAttW);
            attWPreA.assign(attW); // z: Pre Tanh
            attWPreS.assign(a.getActivation(attW, true)); // a: Pre Softmax
            softmax.getActivation(attW, true);

            final INDArray dLdy = Nd4j.gemm(curEps, in, true, false); // 	∂L/∂γ
            final INDArray dLda = softmax.backprop(attWPreS, dLdy).getFirst();// ∂L/∂a
            final INDArray dLdz = a.backprop(attWPreA, dLda).getFirst(); // ∂L/∂z


            // ∂L/∂b
            bg.addi(dLdz.sum());

            // ∂L/∂W
            Wg.addi(Nd4j.gemm(in, dLdz, false, true).sum(1));

            // ∂L/∂Wq
            Wqg.addi(Nd4j.gemm(in, dLdz, false, false).sum(1));

            // ∂L/∂x: Part 1 - from multiplying with attention weight
            curEpsOut.assign(curEps.mmul(attW));
            // ∂L/∂x: Part 2 - from being used in general attention weight calculation
            curEpsOut.addi(W.mmul(dLdz.sum(0)));
            // ∂L/∂x: Part 3 - from being used in query attention weight calculation
            curEpsOut.addi(Wq.mmul(dLdz.sum(1).transposei()));
        }


        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(QueryAttentionParamInitializer.WEIGHT_KEY, Wg);
        g.gradientForVariable().put(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY, Wqg);
        g.gradientForVariable().put(QueryAttentionParamInitializer.BIAS_KEY, bg);

        epsOut = backpropDropOutIfPresent(epsOut);
        return new Pair<>(g, epsOut);
    }
}
