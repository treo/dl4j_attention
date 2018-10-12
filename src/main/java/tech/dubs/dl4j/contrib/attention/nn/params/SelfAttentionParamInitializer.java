/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package tech.dubs.dl4j.contrib.attention.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Paul Dubs
 */
public class SelfAttentionParamInitializer implements ParamInitializer {

    private static final SelfAttentionParamInitializer INSTANCE = new SelfAttentionParamInitializer();

    public static SelfAttentionParamInitializer getInstance(){
        return INSTANCE;
    }

    public static final String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public static final String QUERY_WEIGHT_KEY = "Q";
    public static final String QUERY_KEY = "q";
    public static final String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    private static final List<String> PARAM_KEYS = Collections.unmodifiableList(Arrays.asList(WEIGHT_KEY, QUERY_WEIGHT_KEY, BIAS_KEY, QUERY_KEY));
    private static final List<String> WEIGHT_KEYS = Collections.unmodifiableList(Arrays.asList(WEIGHT_KEY, QUERY_WEIGHT_KEY));
    private static final List<String> BIAS_KEYS = Collections.unmodifiableList(Arrays.asList(BIAS_KEY, QUERY_KEY));


    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer layer) {
        FeedForwardLayer c = (FeedForwardLayer) layer;
        final long nIn = c.getNIn();
        final long nOut = c.getNOut();

        final long paramsW = nIn * nOut;
        final long paramsWq = nIn * nOut;
        final long paramsB = nOut;
        final long paramsQ = nIn;
        return  paramsW + paramsWq + paramsB + paramsQ;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return PARAM_KEYS;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return WEIGHT_KEYS;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return BIAS_KEYS;
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return WEIGHT_KEYS.contains(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEYS.contains(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        FeedForwardLayer c = (FeedForwardLayer) conf.getLayer();
        final long nIn = c.getNIn();
        final long nOut = c.getNOut();

        Map<String,INDArray> m;

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(c.getDist());

            m = getSubsets(paramsView, nIn, nOut, false);
            INDArray w = WeightInitUtil.initWeights(nIn, nOut, new long[]{nIn, nOut}, c.getWeightInit(), dist, 'f', m.get(WEIGHT_KEY));
            m.put(WEIGHT_KEY, w);

            INDArray rq = WeightInitUtil.initWeights(nIn, nOut, new long[]{nIn, nOut}, c.getWeightInit(), dist,'f', m.get(QUERY_WEIGHT_KEY));
            m.put(QUERY_WEIGHT_KEY, rq);

            INDArray q = WeightInitUtil.initWeights(nIn, 1, new long[]{nIn, 1}, c.getWeightInit(), dist,'f', m.get(QUERY_KEY));
            m.put(QUERY_KEY, q);
        } else {
            m = getSubsets(paramsView, nIn, nOut, true);
        }

        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(QUERY_WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);
        conf.addVariable(QUERY_KEY);

        return m;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        FeedForwardLayer c = (FeedForwardLayer) conf.getLayer();
        final long nIn = c.getNIn();
        final long nOut = c.getNOut();

        return getSubsets(gradientView, nIn, nOut, true);
    }

    private static Map<String,INDArray> getSubsets(INDArray in, long nIn, long nOut, boolean reshape){
        final long endW = nIn * nOut;
        final long endWq = endW + nIn * nOut;
        final long endB = endWq + nOut;
        final long endQ = endB + nIn;

        INDArray w = in.get(point(0), interval(0, endW));
        INDArray wq = in.get(point(0), interval(endW, endWq));
        INDArray b = in.get(point(0), interval(endWq, endB));
        INDArray q = in.get(point(0), interval(endB, endQ));

        if (reshape) {
            w = w.reshape('f', nIn, nOut);
            wq = wq.reshape('f', nIn, nOut);
            b = b.reshape('f', 1, nOut);
            q = q.reshape('f', 1, nIn);
        }

        Map<String, INDArray> m = new LinkedHashMap<>();
        m.put(WEIGHT_KEY, w);
        m.put(QUERY_WEIGHT_KEY, wq);
        m.put(BIAS_KEY, b);
        m.put(QUERY_KEY, q);
        return m;
    }
}
