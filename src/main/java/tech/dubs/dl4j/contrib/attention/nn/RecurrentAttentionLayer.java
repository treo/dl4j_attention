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
import org.nd4j.linalg.primitives.Pair;
import tech.dubs.dl4j.contrib.attention.nn.params.RecurrentQueryAttentionParamInitializer;

/**
 * Recurrent Attention Layer Implementation
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
        INDArray attPrep = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{examples, 1, tsLength}, 'f');


        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        // Precalculate recurrency independent parts
        // stepping through example-wise
        final long exampleTads = input.tensorssAlongDimension(1, 2);
        for (int tad = 0; tad < exampleTads; tad++) {
            final INDArray in = input.tensorAlongDimension(tad, 1, 2);
            final INDArray exAttPrep = attPrep.tensorAlongDimension(tad, 1, 2);
            final INDArray z = activations.tensorAlongDimension(tad, 1, 2);

            // For every timestep!
            // Initialize z while at it
            z.assign(W.mmul(in));
            z.addiColumnVector(b);

            exAttPrep.assign(Wq.mmul(in));
            exAttPrep.addi(bq);

            // Recurrent Part
            final long timesteps = in.tensorssAlongDimension(0);
            for (int timestep = 0; timestep < timesteps; timestep++) {
                final INDArray curZ = z.tensorAlongDimension(timestep, 0);

                if(timestep > 0){
                    final INDArray prevA = z.tensorAlongDimension(timestep - 1, 0);

                    final INDArray tsAttW = exAttPrep.dup().addiColumnVector(prevA.mmul(Wqr));
                    a.getActivation(tsAttW, training);
                    softmax.getActivation(tsAttW, training);

                    final INDArray tsAtt = in.mmul(tsAttW.transposei()).transposei();

                    curZ.addi(tsAtt.mmul(Wr));
                }

                a.getActivation(curZ, training);
            }
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

        applyDropOutIfNecessary(true, workspaceMgr);


        long examples = input.size(0);
        long tsLength = input.size(2);
        long nIn = layerConf().getNIn();
        long nOut = layerConf().getNOut();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nOut, tsLength}, 'f');
        INDArray preActivation = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{examples, nOut, tsLength}, 'f');
        INDArray attPerTs = workspaceMgr.create(ArrayType.BP_WORKING_MEM, new long[]{examples, nIn, tsLength}, 'f');

        INDArray attPrep = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{examples, 1, tsLength}, 'f');



        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        // Precalculate recurrency independent parts
        // stepping through example-wise
        final long exampleTads = input.tensorssAlongDimension(1, 2);
        for (int tad = 0; tad < exampleTads; tad++) {
            final INDArray in = input.tensorAlongDimension(tad, 1, 2);
            final INDArray exAttPrep = attPrep.tensorAlongDimension(tad, 1, 2);
            final INDArray z = activations.tensorAlongDimension(tad, 1, 2);
            final INDArray exPreActivation = preActivation.tensorAlongDimension(tad, 1, 2);
            final INDArray exAttPerTs = attPerTs.tensorAlongDimension(tad, 1, 2);

            // For every timestep!
            // Initialize z while at it
            z.assign(W.mmul(in));
            z.addiColumnVector(b);

            exAttPrep.assign(Wq.mmul(in));
            exAttPrep.addi(bq);


            final long timesteps = in.tensorssAlongDimension(0);
            for (int timestep = 0; timestep < timesteps; timestep++) {
                final INDArray curZ = z.tensorAlongDimension(timestep, 0);
                final INDArray curPreActivation = exPreActivation.tensorAlongDimension(timestep, 0);

                if(timestep > 0){
                    final INDArray prevA = z.tensorAlongDimension(timestep - 1, 0);
                    final INDArray curAttPerTs = exAttPerTs.tensorAlongDimension(timestep, 0);

                    final INDArray tsAttW = exAttPrep.dup().addiColumnVector(prevA.mmul(Wqr));
                    a.getActivation(tsAttW, true);
                    softmax.getActivation(tsAttW, true);

                    final INDArray tsAtt = in.mmul(tsAttW.transposei()).transposei();

                    curZ.addi(tsAtt.mmul(Wr));
                }

                curPreActivation.assign(curZ);
                a.getActivation(curZ, true);
            }
        }


        // Backpropagation
        // Only the very last epsilon is complete, all others are missing the recurrent component, so we have to step
        // through it one timestep at a time



        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.WEIGHT_KEY, Wg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.RECURRENT_WEIGHT_KEY, Wrg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.QUERY_WEIGHT_KEY, Wqg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.RECURRENT_QUERY_WEIGHT_KEY, Wqrg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.BIAS_KEY, bg);
        g.gradientForVariable().put(RecurrentQueryAttentionParamInitializer.QUERY_BIAS_KEY, bqg);

        epsOut = backpropDropOutIfPresent(epsOut);

        return new Pair<>(g, epsOut);
    }
}
