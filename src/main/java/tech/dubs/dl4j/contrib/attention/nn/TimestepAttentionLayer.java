package tech.dubs.dl4j.contrib.attention.nn;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.primitives.Pair;
import tech.dubs.dl4j.contrib.attention.activations.ActivationMaskedSoftmax;
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
    private ActivationMaskedSoftmax softmax = new ActivationMaskedSoftmax();

    public TimestepAttentionLayer(NeuralNetConfiguration conf) {
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

        INDArray W = getParamWithNoise(QueryAttentionParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray Q = getParamWithNoise(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(QueryAttentionParamInitializer.BIAS_KEY, training, workspaceMgr);


        long nIn = layerConf().getNIn();
        long nOut = layerConf().getNOut();
        IActivation a = layerConf().getActivationFn();
        long examples = input.shape()[0] == nIn ? input.shape()[2] : input.shape()[0];
        long tsLength = input.shape()[0] == nIn ? input.shape()[1] : input.shape()[2];

        INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{examples, nIn*nOut, tsLength}, 'f');

        if(input.shape()[0] != nIn)
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input.permute(1, 2, 0), 'f');

        final AttentionMechanism attentionMechanism = new AttentionMechanism(Q, W, b, a, workspaceMgr, training);
        final INDArray attention = attentionMechanism.query(input, input, input, maskArray);
        activations.assign(attention);

        return activations;
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if(epsilon.ordering() != 'f' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('f');

        INDArray W = getParamWithNoise(QueryAttentionParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray Q = getParamWithNoise(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY, true, workspaceMgr);
        INDArray b = getParamWithNoise(QueryAttentionParamInitializer.BIAS_KEY, true, workspaceMgr);

        INDArray Wg = gradientViews.get(QueryAttentionParamInitializer.WEIGHT_KEY);
        INDArray Qg = gradientViews.get(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY);
        INDArray bg = gradientViews.get(QueryAttentionParamInitializer.BIAS_KEY);
        gradientsFlattened.assign(0);

        applyDropOutIfNecessary(true, workspaceMgr);

        IActivation a = layerConf().getActivationFn();

        long nIn = layerConf().getNIn();
        if(input.shape()[0] != nIn)
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input.permute(1, 2, 0), 'f');

        INDArray epsOut = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, input.shape(), 'f');

        final AttentionMechanism attentionMechanism = new AttentionMechanism(Q, W, b, a, workspaceMgr, true);
        attentionMechanism
                .withGradientViews(Wg, Qg, bg, epsOut, epsOut, epsOut)
                .backprop(epsilon, input, input, input, maskArray);

        epsOut = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsOut.permute(2, 0, 1), 'f');

        weightNoiseParams.clear();

        Gradient g = new DefaultGradient(gradientsFlattened);
        g.gradientForVariable().put(QueryAttentionParamInitializer.WEIGHT_KEY, Wg);
        g.gradientForVariable().put(QueryAttentionParamInitializer.QUERY_WEIGHT_KEY, Qg);
        g.gradientForVariable().put(QueryAttentionParamInitializer.BIAS_KEY, bg);

        epsOut = backpropDropOutIfPresent(epsOut);
        return new Pair<>(g, epsOut);
    }
}
