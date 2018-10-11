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

package tech.dubs.dl4j.contrib.attention.activations;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * f_i(x, m) = m_i*exp(x_i - shift) / sum_j m_j*exp(x_j - shift)
 * where shift = max_i(x_i), m = mask
 *
 * @author Paul Dubs
 */
public class ActivationMaskedSoftmax {

    public INDArray getActivation(INDArray in, INDArray mask) {
        assertShape(in, mask, null);
        final long[] shape = in.shape();
        final long[] preBroadcastShape = Arrays.copyOf(shape, shape.length);
        preBroadcastShape[preBroadcastShape.length - 1] = 1;

        final INDArray max = in.max(-1).reshape(preBroadcastShape).broadcast(shape);
        final INDArray exp = Transforms.exp(in.sub(max), false);
        final INDArray masked;
        if (mask != null) {
            masked = exp.mul(mask);
        } else {
            masked = exp;
        }

        final INDArray sum = masked.sum(-1).reshape(preBroadcastShape).broadcast(shape);
        masked.div(sum, in);
        return in;
    }

    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray mask, INDArray epsilon) {
        assertShape(in, mask, epsilon);
        INDArray out = getActivation(in, mask);
        INDArray x = out.mul(epsilon).sum(1);
        INDArray dLdz = out.mul(epsilon.subColumnVector(x));
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "maskedSoftmax";
    }

    private void assertShape(INDArray in, INDArray mask, INDArray epsilon) {
        if (mask != null && !in.equalShapes(mask)) {
            throw new IllegalStateException("Shapes must be equal: in.shape{} = " + Arrays.toString(in.shape())
                    + ", mask.shape() = " + Arrays.toString(mask.shape()));
        }
        if (epsilon != null && !in.equalShapes(epsilon)) {
            throw new IllegalStateException("Shapes must be equal during backprop: in.shape{} = " + Arrays.toString(in.shape())
                    + ", epsilon.shape() = " + Arrays.toString(epsilon.shape()));
        }
    }
}
