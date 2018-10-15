package tech.dubs.dl4j.contrib.attention.nn;

import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import tech.dubs.dl4j.contrib.attention.activations.ActivationMaskedSoftmax;

import java.util.Arrays;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/*
 *  Attention: Shapes for keys, values and queries should be in [features, timesteps, examples] order!
 * @author Paul Dubs
 */
public class AdditiveAttentionMechanism {
    private final INDArray W;
    private final INDArray Q;
    private final INDArray b;
    private final IActivation activation;
    private final ActivationMaskedSoftmax softmax;
    private final LayerWorkspaceMgr mgr;
    private final boolean training;
    private boolean caching;
    private INDArray WkCache;

    // Required to be set for backprop
    private INDArray Wg;
    private INDArray Qg;
    private INDArray bg;
    private INDArray keyG;
    private INDArray valueG;
    private INDArray queryG;

    public AdditiveAttentionMechanism(INDArray queryWeight, INDArray keyWeight, INDArray bias, IActivation activation, LayerWorkspaceMgr mgr, boolean training) {
        assertWeightShapes(queryWeight, keyWeight, bias);
        Q = queryWeight;
        W = keyWeight;
        b = bias;
        this.activation = activation;
        softmax = new ActivationMaskedSoftmax();
        this.mgr = mgr;
        this.training = training;

        this.caching = false;
    }

    public AdditiveAttentionMechanism useCaching() {
        this.caching = true;
        return this;
    }

    public INDArray query(INDArray queries, INDArray keys, INDArray values, INDArray mask) {
        assertShapes(queries, keys, values);

        final long examples = queries.shape()[2];
        final long queryCount = queries.shape()[1];
        final long queryWidth = queries.shape()[0];
        final long attentionHeads = W.shape()[1];
        final long memoryWidth = W.shape()[0];
        final long tsLength = keys.shape()[1];

        final INDArray result = mgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{examples, memoryWidth * attentionHeads, queryCount}, 'f');

        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .build();

        if (this.caching && this.WkCache == null) {
            final INDArray target = mgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{attentionHeads, tsLength * examples}, 'f');

            this.WkCache = Nd4j.gemm(W, keys.reshape('f', memoryWidth, tsLength * examples), target, true, false, 1.0, 0.0)
                    .addiColumnVector(b.transpose())
                    .reshape('f', attentionHeads, tsLength, examples);
        }

        final INDArray queryRes = Nd4j.gemm(Q, queries.reshape('f', queryWidth, queryCount * examples), true, false)
                .reshape('f', attentionHeads, queryCount, examples);

        for (long example = 0; example < examples; example++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "ATTENTION_FF")) {
                final INDArray curValues = values.get(all(), all(), point(example));
                final INDArray curKeys = keys.get(all(), all(), point(example));

                final INDArray preResult;
                if (this.caching) {
                    preResult = this.WkCache.get(all(), all(), point(example));
                } else {
                    preResult = Nd4j.gemm(W, curKeys, true, false);
                    preResult.addiColumnVector(b.transpose());
                }


                final INDArray curMask = subMask(mask, example);
                final INDArray attentionHeadMask = attentionHeadMask(curMask, preResult.shape());

                for (long queryIdx = 0; queryIdx < queryCount; queryIdx++) {
                    final INDArray query = queries.get(all(), point(queryIdx), point(example));
                    final INDArray curResult = subArray(result, example, queryIdx);

                    final INDArray queryResult = queryRes.get(all(), point(queryIdx), point(example));

                    final INDArray preA = preResult.addColumnVector(queryResult);
                    final INDArray preS = this.activation.getActivation(preA, training);
                    final INDArray attW = softmax.getActivation(preS, attentionHeadMask);

                    final INDArray att = Nd4j.gemm(curValues, attW, false, true);
                    curResult.assign(att.reshape('f', 1, memoryWidth * attentionHeads));
                }
            }
        }
        return result;
    }

    public AdditiveAttentionMechanism withGradientViews(INDArray W, INDArray Q, INDArray b, INDArray keys, INDArray values, INDArray queries) {
        Wg = W;
        Qg = Q;
        bg = b;
        keyG = keys;
        valueG = values;
        queryG = queries;

        return this;
    }

    public void backprop(INDArray epsilon, INDArray queries, INDArray keys, INDArray values, INDArray mask) {
        if (Wg == null || Qg == null || bg == null || keyG == null || valueG == null || queryG == null) {
            throw new IllegalStateException("You MUST use attnMech.withGradientViews(...).backprop(...).");
        }

        assertShapes(queries, keys, values);

        final long examples = queries.shape()[2];
        final long queryCount = queries.shape()[1];
        final long queryWidth = queries.shape()[0];
        final long attentionHeads = W.shape()[1];
        final long memoryWidth = W.shape()[0];
        final long tsLength = keys.shape()[1];


        if (epsilon.ordering() != 'c' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('c');

        final long[] epsilonShape = epsilon.shape();
        if (epsilonShape[0] != examples || epsilonShape[1] != (attentionHeads * memoryWidth) || (epsilonShape.length == 2 && queryCount != 1) || (epsilonShape.length == 3 && epsilonShape[2] != queryCount)) {
            throw new IllegalStateException("Epsilon shape must match result shape. Got epsilon.shape() = " + Arrays.toString(epsilonShape)
                    + "; result shape = [" + examples + ", " + attentionHeads * memoryWidth + ", " + queryCount + "]");
        }

        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .build();

        final INDArray dldAtt = epsilon.reshape('c', examples, attentionHeads, memoryWidth, queryCount);

        if (this.caching && this.WkCache == null) {
            final INDArray target = mgr.createUninitialized(ArrayType.BP_WORKING_MEM, new long[]{attentionHeads, tsLength * examples}, 'f');

            this.WkCache = Nd4j.gemm(W, keys.reshape('f', memoryWidth, tsLength * examples), target, true, false, 1.0, 0.0)
                    .addiColumnVector(b.transpose())
                    .reshape('f', attentionHeads, tsLength, examples);
        }

        final INDArray queryRes = Nd4j.gemm(Q, queries.reshape('f', queryWidth, queryCount * examples), true, false)
                .reshape('f', attentionHeads, queryCount, examples);

        for (long example = 0; example < examples; example++) {
            try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "ATTENTION_BP")) {
                final INDArray curValues = values.get(all(), all(), point(example));
                final INDArray curKeys = keys.get(all(), all(), point(example));

                final INDArray exEps = dldAtt.tensorAlongDimension((int) example, 1, 2, 3);

                final INDArray preResult;
                if (this.caching) {
                    preResult = this.WkCache.get(all(), all(), point(example));
                } else {
                    preResult = Nd4j.gemm(W, curKeys, true, false);
                    preResult.addiColumnVector(b.transpose());
                }

                final INDArray curMask = subMask(mask, example);
                final INDArray attentionHeadMask = attentionHeadMask(curMask, preResult.shape());

                for (long queryIdx = 0; queryIdx < queryCount; queryIdx++) {
                    final INDArray curEps = exEps.tensorAlongDimension((int) queryIdx, 0, 1);
                    final INDArray query = queries.get(all(), point(queryIdx), point(example));

                    final INDArray queryResult = queryRes.get(all(), point(queryIdx), point(example));

                    final INDArray preA = preResult.addColumnVector(queryResult);
                    final INDArray preS = this.activation.getActivation(preA.dup(), training);
                    final INDArray attW = softmax.getActivation(preS, attentionHeadMask);

                    valueG.get(all(), all(), point(example)).addi(Nd4j.gemm(curEps, attW, true, false));

                    final INDArray dldAttW = Nd4j.gemm(curEps, curValues, false, false);
                    final INDArray dldPreS = softmax.backprop(attW, attentionHeadMask, dldAttW).getFirst();
                    final INDArray dldPreA = activation.backprop(preA, dldPreS).getFirst();

                    final INDArray dldPreASum = dldPreA.sum(1);

                    Nd4j.gemm(query, dldPreASum, Qg, false, true, 1.0, 1.0);
                    Nd4j.gemm(curKeys, dldPreA, Wg, false, true, 1.0, 1.0);

                    bg.addi(dldPreASum.transpose());

                    keyG.get(all(), all(), point(example)).addi(Nd4j.gemm(W, dldPreA, false, false));
                    queryG.get(all(), point(queryIdx), point(example)).addi(Nd4j.gemm(Q, dldPreASum, false, false));
                }
            }
        }
    }

    private void assertWeightShapes(INDArray queryWeight, INDArray keyWeight, INDArray bias) {
        final long qOut = queryWeight.shape()[1];
        final long kOut = keyWeight.shape()[1];
        final long bOut = bias.shape()[1];
        if (qOut != kOut || qOut != bOut) {
            throw new IllegalStateException("Shapes must be compatible: queryWeight.shape() = " + Arrays.toString(queryWeight.shape())
                    + ", keyWeight.shape() = " + Arrays.toString(keyWeight.shape())
                    + ", bias.shape() = " + Arrays.toString(bias.shape())
                    + "\n Compatible shapes should have the same second dimension, but got: [" + qOut + ", " + kOut + ", " + bOut + "]"
            );
        }
    }

    private void assertShapes(INDArray query, INDArray keys, INDArray values) {
        final long kIn = W.shape()[0];
        final long qIn = Q.shape()[0];

        if (query.shape()[0] != qIn || keys.shape()[0] != kIn) {
            throw new IllegalStateException("Shapes of query and keys must be compatible to weights, but got: queryWeight.shape() = " + Arrays.toString(Q.shape())
                    + ", queries.shape() = " + Arrays.toString(query.shape())
                    + "; keyWeight.shape() = " + Arrays.toString(W.shape())
                    + ", keys.shape() = " + Arrays.toString(keys.shape())
            );
        }

        if (keys.shape()[1] != values.shape()[1]) {
            throw new IllegalStateException("Keys must be the same length as values! But got keys.shape() = " + Arrays.toString(keys.shape())
                    + ", values.shape = " + Arrays.toString(values.shape()));
        }

        if (keys.shape()[2] != values.shape()[2] || query.shape()[2] != keys.shape()[2]) {
            throw new IllegalStateException("Queries, Keys and Values must have same mini-batch size! But got keys.shape() = " + Arrays.toString(keys.shape())
                    + ", values.shape = " + Arrays.toString(values.shape())
                    + ", queries.shape = " + Arrays.toString(query.shape())
            );
        }
    }


    private INDArray subArray(INDArray in, long example) {
        return in.tensorAlongDimension((int) example, 1, 2);
    }

    private INDArray subArray(INDArray in, long example, long timestep) {
        return subArray(in, example).tensorAlongDimension((int) timestep, 0);
    }

    private INDArray subMask(INDArray mask, long example) {
        if (mask == null) {
            return null;
        } else {
            return mask.tensorAlongDimension((int) example, 1);
        }
    }

    private INDArray attentionHeadMask(INDArray mask, long[] shape) {
        if (mask == null) {
            return null;
        } else {
            return mask.broadcast(shape);
        }
    }
}
