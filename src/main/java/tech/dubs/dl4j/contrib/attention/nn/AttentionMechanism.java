package tech.dubs.dl4j.contrib.attention.nn;

import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import tech.dubs.dl4j.contrib.attention.activations.ActivationMaskedSoftmax;

import java.util.Arrays;

public class AttentionMechanism {
    private final INDArray W;
    private final INDArray Q;
    private final INDArray b;
    private final IActivation activation;
    private final ActivationMaskedSoftmax softmax;
    private final LayerWorkspaceMgr mgr;
    private final boolean training;

    public AttentionMechanism(INDArray queryWeight, INDArray keyWeight, INDArray bias, IActivation activation, LayerWorkspaceMgr mgr, boolean training){
        assertWeightShapes(queryWeight, keyWeight, bias);
        Q = queryWeight;
        W = keyWeight;
        b = bias;
        this.activation = activation;
        softmax = new ActivationMaskedSoftmax();
        this.mgr = mgr;
        this.training = training;
    }

    public INDArray query(INDArray queries, INDArray keys, INDArray values, INDArray mask){
        assertShapes(queries, keys, values);

        final long examples = queries.shape()[0];
        final long queryCount = queries.shape()[2];
        final long attentionHeads = W.shape()[1];
        final long memoryWidth = values.shape()[1];

        final INDArray result = mgr.createUninitialized(ArrayType.FF_WORKING_MEM, new long[]{examples, memoryWidth * attentionHeads, queryCount}, 'f');

        for (long example = 0; example < examples; example++) {
            final INDArray curValues = subArray(values, example);
            final INDArray curKeys = subArray(keys, example);

            final INDArray preResult = Nd4j.gemm(W, curKeys, true, false);
            preResult.addiColumnVector(b.transpose());

            final INDArray curMask = subMask(mask, example);
            final INDArray attentionHeadMask = attentionHeadMask(curMask, preResult.shape());

            for (long queryIdx = 0; queryIdx < queryCount; queryIdx++) {
                final INDArray query = subArray(queries, example, queryIdx);
                final INDArray curResult = subArray(result, example, queryIdx);

                final INDArray queryResult = Nd4j.gemm(Q, query, true, true);

                final INDArray preA = preResult.addColumnVector(queryResult);
                final INDArray preS = this.activation.getActivation(preA, training);
                final INDArray attW = softmax.getActivation(preS, attentionHeadMask);

                final INDArray att = Nd4j.gemm(curValues, attW, false, true);
                curResult.assign(att.reshape('f', 1, memoryWidth * attentionHeads));
            }
        }

        return result;
    }

    public AttentionGradient backprop(INDArray epsilon, INDArray queries, INDArray keys, INDArray values, INDArray mask){
        assertShapes(queries, keys, values);

        final long examples = queries.shape()[0];
        final long queryCount = queries.shape()[2];
        final long attentionHeads = W.shape()[1];
        final long memoryWidth = values.shape()[1];

        if(epsilon.ordering() != 'c' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('c');

        final long[] epsilonShape = epsilon.shape();
        if(epsilonShape[0] != examples || epsilonShape[1] != (attentionHeads * memoryWidth) || (epsilonShape.length == 2 && queryCount != 1) || (epsilonShape.length == 3 && epsilonShape[2] != queryCount)){
            throw new IllegalStateException("Epsilon shape must match result shape. Got epsilon.shape() = "+Arrays.toString(epsilonShape)
                + "; result shape = ["+examples+", "+attentionHeads*memoryWidth+", "+queryCount+"]");
        }

        final INDArray dldValues = mgr.create(ArrayType.BP_WORKING_MEM, values.shape(), 'f');
        final INDArray dldKeys = mgr.create(ArrayType.BP_WORKING_MEM, keys.shape(), 'f');
        final INDArray dldQueries = mgr.create(ArrayType.BP_WORKING_MEM, queries.shape(), 'f');
        final INDArray dldW = mgr.create(ArrayType.BP_WORKING_MEM, W.shape(), 'f');
        final INDArray dldQ = mgr.create(ArrayType.BP_WORKING_MEM, Q.shape(), 'f');
        final INDArray dldb = mgr.create(ArrayType.BP_WORKING_MEM, b.shape(), 'f');

        final INDArray dldAtt = epsilon.reshape('c', examples, attentionHeads, memoryWidth, queryCount);


        for (long example = 0; example < examples; example++) {
            final INDArray curValues = subArray(values, example);
            final INDArray curKeys = subArray(keys, example);

            final INDArray exEps = dldAtt.tensorAlongDimension((int) example, 1,2,3);

            final INDArray preResult = Nd4j.gemm(W, curKeys, true, false);
            preResult.addiColumnVector(b.transpose());

            final INDArray curMask = subMask(mask, example);
            final INDArray attentionHeadMask = attentionHeadMask(curMask, preResult.shape());

            for (long queryIdx = 0; queryIdx < queryCount; queryIdx++) {
                final INDArray curEps = exEps.tensorAlongDimension((int) queryIdx, 0,1);
                final INDArray query = subArray(queries, example, queryIdx);

                final INDArray queryResult = Nd4j.gemm(Q, query, true, true);

                final INDArray preA = preResult.addColumnVector(queryResult);
                final INDArray preS = this.activation.getActivation(preA.dup(), training);
                final INDArray attW = softmax.getActivation(preS.dup(), attentionHeadMask);

                subArray(dldValues, example).addi(Nd4j.gemm(curEps, attW, true, false));

                final INDArray dldAttW = Nd4j.gemm(curEps, curValues, false, false);
                final INDArray dldPreS = softmax.backprop(preS, attentionHeadMask, dldAttW).getFirst();
                final INDArray dldPreA = activation.backprop(preA, dldPreS).getFirst();

                dldQ.addi(Nd4j.gemm(dldPreA.sum(1), query, false, false).transposei());
                dldW.addi(Nd4j.gemm(curKeys, dldPreA, false, true));

                dldb.addi(dldPreA.sum(1).transpose());

                subArray(dldKeys, example).addi(Nd4j.gemm(W, dldPreA, false, false));
                subArray(dldQueries, example, queryIdx).addi(Nd4j.gemm(Q, dldPreA, false, false).sum(1).transpose());
            }
        }

        return new AttentionGradient(dldW, dldQ, dldb, dldKeys, dldValues, dldQueries);
    }

    private void assertWeightShapes(INDArray queryWeight, INDArray keyWeight, INDArray bias) {
        final long qOut = queryWeight.shape()[1];
        final long kOut = keyWeight.shape()[1];
        final long bOut = bias.shape()[1];
        if(qOut != kOut || qOut != bOut){
            throw new IllegalStateException("Shapes must be compatible: queryWeight.shape() = " + Arrays.toString(queryWeight.shape())
                    + ", keyWeight.shape() = " + Arrays.toString(keyWeight.shape())
                    + ", bias.shape() = " + Arrays.toString(bias.shape())
                    + "\n Compatible shapes should have the same second dimension, but got: ["+qOut+", "+kOut+", "+bOut+"]"
            );
        }
    }

    private void assertShapes(INDArray query, INDArray keys, INDArray values) {
        final long kIn = W.shape()[0];
        final long qIn = Q.shape()[0];

        if(query.shape()[1] != qIn || keys.shape()[1] != kIn){
            throw new IllegalStateException("Shapes of query and keys must be compatible to weights, but got: queryWeight.shape() = "+Arrays.toString(Q.shape())
                    +", queries.shape() = "+Arrays.toString(query.shape())
                    +"; keyWeight.shape() = " + Arrays.toString(W.shape())
                    +", keys.shape() = " + Arrays.toString(keys.shape())
            );
        }

        if(keys.shape()[2] != values.shape()[2]){
            throw new IllegalStateException("Keys must be the same length as values! But got keys.shape() = "+Arrays.toString(keys.shape())
                    +", values.shape = "+Arrays.toString(values.shape()));
        }

        if(keys.shape()[0] != values.shape()[0] || query.shape()[0] != keys.shape()[0]){
            throw new IllegalStateException("Queries, Keys and Values must have same mini-batch size! But got keys.shape() = "+Arrays.toString(keys.shape())
                    +", values.shape = "+Arrays.toString(values.shape())
                    +", queries.shape = "+Arrays.toString(query.shape())
            );
        }
    }



    private INDArray subArray(INDArray in, long example){
        return in.tensorAlongDimension((int) example, 1, 2);
    }

    private INDArray subArray(INDArray in, long example, long timestep){
        return subArray(in, example).tensorAlongDimension((int) timestep, 0);
    }

    private INDArray subMask(INDArray mask, long example){
        if(mask == null){
            return null;
        }else{
            return mask.tensorAlongDimension((int) example, 1);
        }
    }

    private INDArray attentionHeadMask(INDArray mask, long[] shape){
        if(mask == null){
            return null;
        }else{
            return mask.broadcast(shape);
        }
    }


    public static class AttentionGradient {

        private final INDArray w;
        private final INDArray q;
        private final INDArray b;
        private final INDArray keys;
        private final INDArray values;
        private final INDArray queries;

        public AttentionGradient(INDArray W, INDArray Q, INDArray b, INDArray keys, INDArray values, INDArray queries){
            this.w = W;
            this.q = Q;
            this.b = b;
            this.keys = keys;
            this.values = values;
            this.queries = queries;
        }

        public INDArray getW() {
            return w;
        }

        public INDArray getQ() {
            return q;
        }

        public INDArray getB() {
            return b;
        }

        public INDArray getKeys() {
            return keys;
        }

        public INDArray getValues() {
            return values;
        }

        public INDArray getQueries() {
            return queries;
        }

        @Override
        public String toString() {
            return "AttentionGradient{\n" +
                    "w=\n" + w +
                    ",\n q=\n" + q +
                    ",\n b=\n" + b +
                    ",\n keys=\n" + keys +
                    ",\n values=\n" + values +
                    ",\n queries=\n" + queries +
                    "\n}";
        }
    }
}
