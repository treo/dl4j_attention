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
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import tech.dubs.dl4j.contrib.attention.nn.params.RecurrentQueryAttentionParamInitializer;

/**
 * Recurrent Attention Layer Implementation
 *
 *
 *
 * TODO:
 *  - Optionally keep attention weights around for inspection
 *  - Handle Masking
 *
 * @author Paul Dubs
 */
public class RecurrentAttentionLayer extends BaseLayer<tech.dubs.dl4j.contrib.attention.conf.RecurrentAttentionLayer> {
    private IActivation softmax = new ActivationSoftmax();

    public RecurrentAttentionLayer(NeuralNetConfiguration conf) {
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

        INDArray W = getParamWithNoise(RecurrentQueryAttentionParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray Wr = getParamWithNoise(RecurrentQueryAttentionParamInitializer.RECURRENT_WEIGHT_KEY, training, workspaceMgr);
        INDArray Wq = getParamWithNoise(RecurrentQueryAttentionParamInitializer.QUERY_WEIGHT_KEY, training, workspaceMgr);
        INDArray Wqr = getParamWithNoise(RecurrentQueryAttentionParamInitializer.RECURRENT_QUERY_WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(RecurrentQueryAttentionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray bq = getParamWithNoise(RecurrentQueryAttentionParamInitializer.QUERY_BIAS_KEY, training, workspaceMgr);

        long examples = input.size(0);
        long tsLength = input.size(2);
        long nIn = layerConf().getNIn();
        long nOut = layerConf().getNOut();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nOut, tsLength}, 'f');

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');


        final AttentionMechanism attentionMechanism = new AttentionMechanism(Wqr, Wq, bq, a, workspaceMgr, training);

        // pre-compute non-recurrent part
        activations.assign(input
                .permute(0,2,1).reshape(examples * tsLength, nIn)
                .mmul(W).addiRowVector(b)
                .reshape(examples, tsLength, nOut).permute(0,2,1));


        for (long timestep = 0; timestep < tsLength; timestep++) {
            final INDArray curOut = timestepArray(activations, timestep);
            if(timestep > 0){
                final INDArray prevActivation = timestepArray(activations, timestep - 1);
                final INDArray attention = attentionMechanism.query(Nd4j.expandDims(prevActivation, 2), input, input, maskArray);
                curOut.addi(Nd4j.squeeze(attention, 2).mmul(Wr));
            }

            a.getActivation(curOut, true);
        }


        return activations;
    }



    /*
     * Notice that the epsilon given here does not contain the recurrent component, which will have to be calculated
     * manually.
     */
    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if(epsilon.ordering() != 'f' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('f');

        INDArray W = getParamWithNoise(RecurrentQueryAttentionParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray Wr = getParamWithNoise(RecurrentQueryAttentionParamInitializer.RECURRENT_WEIGHT_KEY, true, workspaceMgr);
        INDArray Wq = getParamWithNoise(RecurrentQueryAttentionParamInitializer.QUERY_WEIGHT_KEY, true, workspaceMgr);
        INDArray Wqr = getParamWithNoise(RecurrentQueryAttentionParamInitializer.RECURRENT_QUERY_WEIGHT_KEY, true, workspaceMgr);
        INDArray b = getParamWithNoise(RecurrentQueryAttentionParamInitializer.BIAS_KEY, true, workspaceMgr);
        INDArray bq = getParamWithNoise(RecurrentQueryAttentionParamInitializer.QUERY_BIAS_KEY, true, workspaceMgr);

        INDArray Wg = gradientViews.get(RecurrentQueryAttentionParamInitializer.WEIGHT_KEY);
        INDArray Wrg = gradientViews.get(RecurrentQueryAttentionParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray Wqg = gradientViews.get(RecurrentQueryAttentionParamInitializer.QUERY_WEIGHT_KEY);
        INDArray Wqrg = gradientViews.get(RecurrentQueryAttentionParamInitializer.RECURRENT_QUERY_WEIGHT_KEY);
        INDArray bg = gradientViews.get(RecurrentQueryAttentionParamInitializer.BIAS_KEY);
        INDArray bqg = gradientViews.get(RecurrentQueryAttentionParamInitializer.QUERY_BIAS_KEY);
        gradientsFlattened.assign(0);

        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.shape(), 'f');
        epsOut.assign(0);

        applyDropOutIfNecessary(true, workspaceMgr);


        long examples = input.size(0);
        long tsLength = input.size(2);
        long nIn = layerConf().getNIn();
        long nOut = layerConf().getNOut();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nOut, tsLength}, 'f');
        INDArray preOut = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{examples, nOut, tsLength}, 'f');
        INDArray attentions = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{examples, nIn, tsLength}, 'f');
        INDArray queryG = workspaceMgr.create(ArrayType.BP_WORKING_MEM, new long[]{examples, nOut, 1}, 'f');

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input)) {
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');
        }


        final AttentionMechanism attentionMechanism = new AttentionMechanism(Wqr, Wq, bq, a, workspaceMgr, true);

        // pre-compute non-recurrent part
        activations.assign(input
                .permute(0,2,1).reshape(examples * tsLength, nIn)
                .mmul(W).addiRowVector(b)
                .reshape(examples, tsLength, nOut).permute(0,2,1));



        for (long timestep = 0; timestep < tsLength; timestep++) {
            final INDArray curOut = timestepArray(activations, timestep);

            if(timestep > 0){
                final INDArray prevActivation = timestepArray(activations, timestep - 1);
                final INDArray query = Nd4j.expandDims(prevActivation, 2);
                final INDArray attention = Nd4j.squeeze(attentionMechanism.query(query, input, input, maskArray), 2);
                timestepArray(attentions, timestep).assign(attention);

                curOut.addi(attention.mmul(Wr));
            }
            timestepArray(preOut, timestep).assign(curOut);
            a.getActivation(curOut, true);
        }


        for (long timestep = tsLength - 1; timestep >= 0; timestep--) {
            final INDArray curEps = timestepArray(epsilon, timestep);
            final INDArray curPreOut = timestepArray(preOut, timestep);
            final INDArray curIn = timestepArray(input, timestep);

            final INDArray dldz = a.backprop(curPreOut, curEps).getFirst();
            Wg.addi(Nd4j.gemm(curIn, dldz, true, false));
            bg.addi(dldz.sum(0));
            timestepArray(epsOut, timestep).addi(Nd4j.gemm(dldz, W, false, true));

            if(timestep > 0){
                final INDArray curAttn = timestepArray(attentions, timestep);

                Wrg.addi(Nd4j.gemm(curAttn, dldz, true, false));

                final INDArray prevEps = timestepArray(epsilon, timestep - 1);
                final INDArray prevActivation = timestepArray(activations, timestep - 1);
                final INDArray query = Nd4j.expandDims(prevActivation, 2);
                queryG.assign(0);

                final INDArray dldAtt = Nd4j.gemm(dldz, Wr, false, true);
                attentionMechanism
                        .withGradientViews(Wqg, Wqrg, bqg, epsOut, epsOut, queryG)
                        .backprop(dldAtt, query, input, input, maskArray);

                prevEps.addi(Nd4j.squeeze(queryG, 2));
            }
        }

        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.WEIGHT_KEY, Wg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.QUERY_WEIGHT_KEY, Wqg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.RECURRENT_QUERY_WEIGHT_KEY, Wqrg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.RECURRENT_WEIGHT_KEY, Wrg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.BIAS_KEY, bg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.QUERY_BIAS_KEY, bqg);

        epsOut = backpropDropOutIfPresent(epsOut);

        return new Pair<>(g, epsOut);
    }

    private INDArray subArray(INDArray in, int example, int timestep){
        return in.tensorAlongDimension(example, 1, 2).tensorAlongDimension(timestep, 0);
    }

    private INDArray timestepArray(INDArray in, long timestep){
        return in.tensorAlongDimension((int) timestep, 0, 1);
    }
}
