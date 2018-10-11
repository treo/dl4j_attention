package tech.dubs.dl4j.contrib.attention.nn;

import org.deeplearning4j.nn.api.MaskState;
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
import tech.dubs.dl4j.contrib.attention.nn.params.SelfAttentionParamInitializer;

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

        INDArray W = getParamWithNoise(SelfAttentionParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray Q = getParamWithNoise(SelfAttentionParamInitializer.QUERY_WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(SelfAttentionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray q = getParamWithNoise(SelfAttentionParamInitializer.QUERY_KEY, training, workspaceMgr);

        long examples = input.size(0);
        long nOut = layerConf().getNOut();
        long nIn = layerConf().getNIn();
        IActivation a = layerConf().getActivationFn();

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nIn * nOut}, 'f');

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        final AttentionMechanism attentionMechanism = new AttentionMechanism(Q, W, b, a, workspaceMgr, training);
        final INDArray attention = attentionMechanism.query(q.reshape(1, nIn, 1).broadcast(examples, nIn, 1), input, input, maskArray);
        activations.assign(attention.reshape(activations.shape()));

        return activations;
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        INDArray W = getParamWithNoise(SelfAttentionParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray Q = getParamWithNoise(SelfAttentionParamInitializer.QUERY_WEIGHT_KEY, true, workspaceMgr);
        INDArray b = getParamWithNoise(SelfAttentionParamInitializer.BIAS_KEY, true, workspaceMgr);
        INDArray q = getParamWithNoise(SelfAttentionParamInitializer.QUERY_KEY, true, workspaceMgr);

        INDArray Wg = gradientViews.get(SelfAttentionParamInitializer.WEIGHT_KEY);
        INDArray Qg = gradientViews.get(SelfAttentionParamInitializer.QUERY_WEIGHT_KEY);
        INDArray bg = gradientViews.get(SelfAttentionParamInitializer.BIAS_KEY);
        INDArray qg = gradientViews.get(SelfAttentionParamInitializer.QUERY_KEY);

        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.shape(), 'f');

        applyDropOutIfNecessary(true, workspaceMgr);

        long examples = input.size(0);
        long nIn = layerConf().getNIn();
        IActivation a = layerConf().getActivationFn();

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');

        final AttentionMechanism attentionMechanism = new AttentionMechanism(Q, W, b, a, workspaceMgr, true);
        final AttentionMechanism.AttentionGradient ag = attentionMechanism.backprop(epsilon, q.reshape(1, nIn, 1).broadcast(examples, nIn, 1), input, input, maskArray);

        epsOut.assign(ag.getKeys()).addi(ag.getValues());
        Wg.assign(ag.getW());
        Qg.assign(ag.getQ());
        bg.assign(ag.getB());
        qg.assign(ag.getQueries().sum(0).transpose());

        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(SelfAttentionParamInitializer.WEIGHT_KEY, Wg);
        g.gradientForVariable().put(SelfAttentionParamInitializer.QUERY_WEIGHT_KEY, Qg);
        g.gradientForVariable().put(SelfAttentionParamInitializer.BIAS_KEY, bg);
        g.gradientForVariable().put(SelfAttentionParamInitializer.QUERY_KEY, qg);

        epsOut = backpropDropOutIfPresent(epsOut);
        return new Pair<>(g, epsOut);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        // no masking is possible after this point... i.e., masks have been taken into account
        // as part of the selfattention
        this.maskArray = maskArray;
        this.maskState = null; //Not used in global pooling - always applied

        return null;
    }
}
