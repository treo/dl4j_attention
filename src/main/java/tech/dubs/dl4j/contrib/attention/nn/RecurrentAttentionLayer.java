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

        // stepping through example-wise and timestep wise
        for (int example = 0; example < examples; example++) {
            final INDArray x = input.tensorAlongDimension(example, 1, 2);
            for (int ts = 0; ts < tsLength; ts++) {
                final INDArray x_i = input.tensorAlongDimension(example, 1, 2).tensorAlongDimension(ts, 0);
                final INDArray z_i = activations.tensorAlongDimension(example, 1, 2).tensorAlongDimension(ts, 0);

                z_i.assign(x_i.mmul(W).addi(b));
                if(ts > 0){
                    final INDArray a_p = activations.tensorAlongDimension(example, 1, 2).tensorAlongDimension(ts - 1, 0);

                    final INDArray tsAttW_PreA = Wq.transpose().mmul(x).addi(bq).addiColumnVector(a_p.mmul(Wqr));
                    final INDArray tsAttW_PreS = a.getActivation(tsAttW_PreA.dup(), training);
                    final INDArray tsAttW = softmax.getActivation(tsAttW_PreS.dup(), training);

                    final INDArray att = x.mmul(tsAttW.transpose());

                    z_i.addi(att.transpose().mmul(Wr));
                }
                a.getActivation(z_i, training);
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
        epsOut.assign(0);

        applyDropOutIfNecessary(true, workspaceMgr);


        long examples = input.size(0);
        long tsLength = input.size(2);
        long nIn = layerConf().getNIn();
        long nOut = layerConf().getNOut();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nOut, tsLength}, 'f');
        INDArray preOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nOut, tsLength}, 'f');
        INDArray attW_PreA = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, tsLength, tsLength}, 'f');
        INDArray attW_PreS = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, tsLength, tsLength}, 'f');
        INDArray attW = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, tsLength, tsLength}, 'f');
        INDArray attentions = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nIn, tsLength}, 'f');


        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        // stepping through example-wise and timestep wise
        for (int example = 0; example < examples; example++) {
            final INDArray x = input.tensorAlongDimension(example, 1, 2);
            for (int ts = 0; ts < tsLength; ts++) {
                final INDArray x_i = subArray(input, example, ts);
                final INDArray z_i = subArray(activations, example, ts);
                final INDArray preOut_i = subArray(preOut, example, ts);

                z_i.assign(x_i.mmul(W).addi(b));
                if(ts > 0){
                    final INDArray a_p = subArray(activations, example, ts - 1);

                    final INDArray tsAttW_PreA = Wq.transpose().mmul(x).addi(bq).addiColumnVector(a_p.mmul(Wqr));
                    subArray(attW_PreA, example, ts).assign(tsAttW_PreA);

                    final INDArray tsAttW_PreS = a.getActivation(tsAttW_PreA.dup(), true);
                    subArray(attW_PreS, example, ts).assign(tsAttW_PreS);

                    final INDArray tsAttW = softmax.getActivation(tsAttW_PreS.dup(), true);
                    subArray(attW, example, ts).assign(tsAttW);

                    final INDArray att = x.mmul(tsAttW.transpose()).transposei();
                    subArray(attentions, example, ts).assign(att);

                    z_i.addi(att.mmul(Wr));
                }
                preOut_i.assign(z_i);
                a.getActivation(z_i, true);
            }
        }


        // Backpropagation
        // Only the very last epsilon is complete, all others are missing the recurrent component, so we have to step
        // through it one timestep at a time
        for (int example = 0; example < examples; example++) {
            final INDArray x = input.tensorAlongDimension(example, 1, 2);
            final INDArray exEpsOut = epsOut.tensorAlongDimension(example, 1, 2);
            for (int ts = (int)tsLength - 1; ts >= 0; ts--) {
                final INDArray x_i = subArray(input, example, ts);
                final INDArray epsOut_i = subArray(epsOut, example, ts);

                final INDArray eps_i = subArray(epsilon, example, ts);
                final INDArray preOut_i = subArray(preOut, example, ts);

                final INDArray dldz = a.backprop(preOut_i, eps_i).getFirst();

                final INDArray dldx_i = dldz.mmul(W.transpose());
                epsOut_i.addi(dldx_i);

                final INDArray dldW = x_i.transpose().mmul(dldz);
                Wg.addi(dldW);

                bg.addi(dldz);

                if(ts > 0){
                    final INDArray att_i = subArray(attentions, example, ts);
                    final INDArray dldWr = att_i.transpose().mmul(dldz);
                    Wrg.addi(dldWr);

                    final INDArray dldAtt = dldz.mmul(Wr.transpose());

                    final INDArray attW_i = subArray(attW, example, ts);
                    final INDArray dldx_1 = dldAtt.transpose().mmul(attW_i);
                    exEpsOut.addi(dldx_1);

                    final INDArray dldAttW = dldAtt.mmul(x);
                    final INDArray dldPreS = softmax.backprop(subArray(attW_PreS, example, ts), dldAttW).getFirst();
                    final INDArray dldPreA = a.backprop(subArray(attW_PreA, example, ts), dldPreS).getFirst();

                    final INDArray dldbq = dldPreA.sum(1);
                    bqg.addi(dldbq);

                    final INDArray dldx_2 = Wq.mmul(dldPreA);
                    exEpsOut.addi(dldx_2);

                    final INDArray dldWq = x.mmul(dldPreA.transpose());
                    Wqg.addi(dldWq);

                    final INDArray a_p = subArray(activations, example, ts - 1);
                    final INDArray dldWqr = a_p.transpose().mmul(dldPreA).sum(1);
                    Wqrg.addi(dldWqr);


                    final INDArray dlda_p = Wqr.mmul(dldPreA.sum(1));
                    subArray(epsilon, example, ts - 1).addi(dlda_p.transposei());
                }
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
}
