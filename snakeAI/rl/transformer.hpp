#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP
#include "attention.hpp"
#include "layer.h"
#include "ilayer.h"

namespace RL {

/*
    TransformerBlock<NumHeads, d_ff> — Standard Pre-LN Transformer Block

    Architecture (Pre-LN):
        x_norm1 = LayerNorm(x)
        attn_out = MultiHeadAttention<NumHeads>(x_norm1)
        x_res1 = x + attn_out                      # Residual 1

        x_norm2 = LayerNorm(x_res1)
        ffn_hidden = GeLU(W_up · x_norm2 + b_up)  # FFN Up (d_model → d_ff)
        ffn_out = W_down · ffn_hidden + b_down     # FFN Down (d_ff → d_model)
        o = x_res1 + ffn_out                        # Residual 2

    Key design points:
        - Pre-LayerNorm (more stable than Post-LN)
        - GeLU activation in FFN
        - Standard output projection Wo in MHA
        - All operations are element-wise residual connections

    Type registration: LAYER_TRANSFORMERBLOCK
*/
template<int NumHeads, int d_ff = 0>
class TransformerBlock : public iLayer
{
public:
    struct NormGrad {
        Tensor gamma;
        Tensor beta;
        NormGrad(){}
        void zero() { gamma.zero(); beta.zero(); }
    };

    int d_model;             // model dimension
    int d_ff_;               // FFN hidden dimension (defaults to 4*d_model)

    /* LayerNorm 1 (pre-attention) parameters */
    Tensor gamma1;           // scale (d_model × 1)
    Tensor beta1;            // shift (d_model × 1)
    NormGrad g1, v1, m1;

    /* Multi-Head Attention */
    MultiHeadAttention<NumHeads> attn;

    /* LayerNorm 2 (pre-FFN) parameters */
    Tensor gamma2;           // scale (d_model × 1)
    Tensor beta2;            // shift (d_model × 1)
    NormGrad g2, v2, m2;

    /* FFN layers */
    Layer<Gelu> ffn_up;      // d_model → d_ff
    Layer<Linear> ffn_down;  // d_ff → d_model

    /* Cached intermediate values for backward */
    //Tensor x_orig;           // original input x
    Tensor x_norm1;          // normalized x (pre-attention norm)
    float mu1, sig1;         // mean + std for norm1
    Tensor x_res1;           // after residual 1: x + attn_out
    Tensor x_norm2;          // normalized x_res1 (pre-FFN norm)
    float mu2, sig2;         // mean + std for norm2
    Tensor ffn_hidden;       // GeLU activation output of ffn_up

public:
    TransformerBlock() {}
    virtual ~TransformerBlock() {}

    static std::shared_ptr<TransformerBlock> _(
        int d_model_, bool withGrad)
    {
        return std::make_shared<TransformerBlock>(d_model_, withGrad);
    }

    explicit TransformerBlock(int d_model_, bool withGrad)
        : d_model(d_model_)
    {
        type = LAYER_TRANSFORMERBLOCK;
        d_ff_ = (d_ff > 0) ? d_ff : 4 * d_model;

        /* LayerNorm parameters */
        gamma1 = Tensor(d_model, 1);
        beta1  = Tensor(d_model, 1);
        Random::uniform(gamma1, 0.9, 1.1);
        Random::uniform(beta1, -0.1, 0.1);

        gamma2 = Tensor(d_model, 1);
        beta2  = Tensor(d_model, 1);
        Random::uniform(gamma2, 0.9, 1.1);
        Random::uniform(beta2, -0.1, 0.1);

        /* Multi-Head Attention */
        attn = MultiHeadAttention<NumHeads>(d_model, d_model, withGrad);

        /* FFN layers */
        ffn_up   = Layer<Gelu>(d_model, d_ff_, true, withGrad);
        ffn_down = Layer<Linear>(d_ff_, d_model, true, withGrad);

        /* Output and error */
        o = Tensor(d_model, 1);
        e = Tensor(d_model, 1);

        /* Cached intermediates */
        //x_orig    = Tensor(d_model, 1);
        x_norm1   = Tensor(d_model, 1);
        x_res1    = Tensor(d_model, 1);
        x_norm2   = Tensor(d_model, 1);
        ffn_hidden = Tensor(d_ff_, 1);

        /* Gradient tensors */
        if (withGrad) {
            g1.gamma = Tensor(d_model, 1);
            g1.beta  = Tensor(d_model, 1);
            v1.gamma = Tensor(d_model, 1);
            v1.beta  = Tensor(d_model, 1);
            m1.gamma = Tensor(d_model, 1);
            m1.beta  = Tensor(d_model, 1);

            g2.gamma = Tensor(d_model, 1);
            g2.beta  = Tensor(d_model, 1);
            v2.gamma = Tensor(d_model, 1);
            v2.beta  = Tensor(d_model, 1);
            m2.gamma = Tensor(d_model, 1);
            m2.beta  = Tensor(d_model, 1);
        }
    }

    /* ==================== Forward ==================== */

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        /* Save original input for backward */
        //x_orig = x;

        /* LayerNorm 1: x_norm1 = gamma1 * (x - mu1) / sig1 + beta1 */
        mu1 = x.mean();
        float var1 = x.variance(mu1);
        sig1 = std::sqrt(var1 + 1e-9f);
        for (int i = 0; i < d_model; i++) {
            x_norm1[i] = gamma1[i] * (x[i] - mu1) / sig1 + beta1[i];
        }

        /* Multi-Head Attention */
        Tensor &attn_out = attn.forward(x_norm1, inference);

        /* Residual 1: x_res1 = x + attn_out */
        for (int i = 0; i < d_model; i++) {
            x_res1[i] = x[i] + attn_out[i];
        }

        /* LayerNorm 2: x_norm2 = gamma2 * (x_res1 - mu2) / sig2 + beta2 */
        mu2 = x_res1.mean();
        float var2 = x_res1.variance(mu2);
        sig2 = std::sqrt(var2 + 1e-9f);
        for (int i = 0; i < d_model; i++) {
            x_norm2[i] = gamma2[i] * (x_res1[i] - mu2) / sig2 + beta2[i];
        }

        /* FFN Up: ffn_hidden = GeLU(W_up · x_norm2 + b_up) */
        ffn_up.forward(x_norm2, inference);
        ffn_hidden = ffn_up.o;  // cache the GeLU output

        /* FFN Down: ffn_out = W_down · ffn_hidden + b_down */
        ffn_down.forward(ffn_hidden, inference);
        Tensor &ffn_out = ffn_down.o;

        /* Residual 2: o = x_res1 + ffn_out */
        for (int i = 0; i < d_model; i++) {
            o[i] = x_res1[i] + ffn_out[i];
        }

        return o;
    }

    /* ==================== Backward ==================== */

    /*
        Unified backward pass for TransformerBlock.

        Gradient flow (Pre-LN Transformer):

            o = x_res1 + ffn_out                            (Residual 2)
            ffn_out = W_down · GeLU(W_up · x_norm2 + b_up) + b_down   (FFN)
            x_norm2 = LayerNorm(x_res1, gamma2, beta2)      (LN 2)
            x_res1 = x_orig + attn_out                      (Residual 1)
            attn_out = Wo · concat(heads(x_norm1))          (MHA)
            x_norm1 = LayerNorm(x_orig, gamma1, beta1)      (LN 1)

        Given e = ∂L/∂o:
          Step 1: Residual 2: d_x_res1 = e
          Step 2: FFN backward via Layer::backward (computes param grads + d_x_norm2)
          Step 3: LN2 backward: accumulates d_x_res1
          Step 4: MHA backward via MultiHeadAttention::backward (computes param grads + d_x_norm1)
          Step 5: LN1 backward: accumulates into ei
    */
    void backward(const Tensor& x, Tensor &ei) override
    {
        float inv_N = 1.0f / d_model;

        /* === Step 1: Residual 2 — dL/d_x_res1 starts as e === */
        Tensor d_x_res1 = e;  // shallow copy (d_model x 1)

        /* === Step 2: FFN backward (via Layer::backward for param grads) === */
        /* ffn_down (Layer<Linear>): d_ffn_hidden = W_down^T · e */
        Tensor d_hidden(d_ff_, 1);
        d_hidden.zero();
        ffn_down.e = e;
        ffn_down.backward(ffn_hidden, d_hidden);
        /* d_hidden now = W_down^T · e (Linear, no activation derivative) */

        /* ffn_up (Layer<Gelu>): applies Gelu derivative internally */
        Tensor d_x_norm2(d_model, 1);
        d_x_norm2.zero();
        ffn_up.e = d_hidden;
        ffn_up.backward(x_norm2, d_x_norm2);
        /* d_x_norm2 now = W_up^T · [Gelu'(pre_act) · W_down^T · e] */

        /* === Step 3: LayerNorm 2 backward (accumulates into d_x_res1) === */
        {
            float inv_sig2 = 1.0f / sig2;
            float sum_dy = 0, sum_dy_dx = 0;
            for (int i = 0; i < d_model; i++) {
                float x_hat = (x_res1[i] - mu2) * inv_sig2;
                sum_dy    += d_x_norm2[i];
                sum_dy_dx += d_x_norm2[i] * x_hat;
            }
            float mean_dy     = sum_dy * inv_N;
            float mean_dy_dx  = sum_dy_dx * inv_N;
            for (int i = 0; i < d_model; i++) {
                float x_hat = (x_res1[i] - mu2) * inv_sig2;
                d_x_res1[i] += gamma2[i] * inv_sig2
                               * (d_x_norm2[i] - mean_dy - x_hat * mean_dy_dx);
            }

            /* Gamma2, Beta2 parameter gradients */
            for (int i = 0; i < d_model; i++) {
                float x_hat = (x_res1[i] - mu2) * inv_sig2;
                g2.gamma[i] += d_x_norm2[i] * x_hat;
                g2.beta[i]  += d_x_norm2[i];
            }
        }

        /* === Step 4: MHA backward (delegate to MultiHeadAttention::backward) === */
        /*
            x_res1 = x_orig + attn_out, so dL/d_attn_out = d_x_res1.
            attn.backward correctly uses e (which we set to d_x_res1) to:
              - Compute da = Wo^T · d_x_res1 and distribute to heads
              - Backprop each head's ScaledDotProduct into d_x_norm1
              - Compute g.wo += d_x_res1 · a^T
        */
        Tensor d_x_norm1(d_model, 1);
        d_x_norm1.zero();
        attn.e = d_x_res1;   // ∂L/∂attn_out = d_x_res1
        attn.backward(x_norm1, d_x_norm1);
        attn.a.zero();        // clear cached head outputs

        /* === Step 5: LayerNorm 1 backward + accumulate to ei === */
        /*
            d_x_res1 also contains the residual path gradient.
            LN1 backward computes: dL/d_x_orig += gamma1/sig1 · (d_x_norm1 - mean - x_hat·mean_dx)
            Total: ei += d_x_res1 (residual) + LN1(d_x_norm1)
        */
        {
            float inv_sig1 = 1.0f / sig1;
            float sum_dy = 0, sum_dy_dx = 0;
            for (int i = 0; i < d_model; i++) {
                float x_hat = (x[i] - mu1) * inv_sig1;
                sum_dy    += d_x_norm1[i];
                sum_dy_dx += d_x_norm1[i] * x_hat;
            }
            float mean_dy    = sum_dy * inv_N;
            float mean_dy_dx = sum_dy_dx * inv_N;
            for (int i = 0; i < d_model; i++) {
                float x_hat = (x[i] - mu1) * inv_sig1;
                ei[i] += d_x_res1[i];  // residual path
                ei[i] += gamma1[i] * inv_sig1
                         * (d_x_norm1[i] - mean_dy - x_hat * mean_dy_dx);  // LN path
            }

            /* Gamma1, Beta1 parameter gradients */
            for (int i = 0; i < d_model; i++) {
                float x_hat = (x[i] - mu1) * inv_sig1;
                g1.gamma[i] += d_x_norm1[i] * x_hat;
                g1.beta[i]  += d_x_norm1[i];
            }
        }

        /* Reset all cached intermediate values */
        o.zero();
        e.zero();
        x_norm1.zero();
        x_res1.zero();
        x_norm2.zero();
        ffn_hidden.zero();
        return;
    }

    /* ==================== Optimizers ==================== */

    void SGD(float lr) override
    {
        Optimize::SGD(gamma1, g1.gamma, lr);
        Optimize::SGD(beta1,  g1.beta,  lr);
        Optimize::SGD(gamma2, g2.gamma, lr);
        Optimize::SGD(beta2,  g2.beta,  lr);
        attn.SGD(lr);
        ffn_up.SGD(lr);
        ffn_down.SGD(lr);
        g1.zero(); g2.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(gamma1, v1.gamma, g1.gamma, lr, rho, decay, clipGrad);
        Optimize::RMSProp(beta1,  v1.beta,  g1.beta,  lr, rho, decay, clipGrad);
        Optimize::RMSProp(gamma2, v2.gamma, g2.gamma, lr, rho, decay, clipGrad);
        Optimize::RMSProp(beta2,  v2.beta,  g2.beta,  lr, rho, decay, clipGrad);
        attn.RMSProp(lr, rho, decay, clipGrad);
        ffn_up.RMSProp(lr, rho, decay, clipGrad);
        ffn_down.RMSProp(lr, rho, decay, clipGrad);
        g1.zero(); g2.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(gamma1, v1.gamma, m1.gamma, g1.gamma,
                       alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(beta1,  v1.beta,  m1.beta,  g1.beta,
                       alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(gamma2, v2.gamma, m2.gamma, g2.gamma,
                       alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        Optimize::Adam(beta2,  v2.beta,  m2.beta,  g2.beta,
                       alpha_, beta_, lr, alpha, beta, decay, clipGrad);
        attn.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        ffn_up.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        ffn_down.Adam(lr, alpha, beta, alpha_, beta_, decay, clipGrad);
        g1.zero(); g2.zero();
        return;
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(gamma1, c0, cn);
        Optimize::clamp(beta1,  c0, cn);
        Optimize::clamp(gamma2, c0, cn);
        Optimize::clamp(beta2,  c0, cn);
        attn.clamp(c0, cn);
        ffn_up.clamp(c0, cn);
        ffn_down.clamp(c0, cn);
        return;
    }

    void copyTo(iLayer* layer) override
    {
        TransformerBlock *p = static_cast<TransformerBlock*>(layer);
        p->gamma1 = gamma1;  p->beta1 = beta1;
        p->gamma2 = gamma2;  p->beta2 = beta2;
        attn.copyTo(&p->attn);
        ffn_up.copyTo(&p->ffn_up);
        ffn_down.copyTo(&p->ffn_down);
        return;
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        TransformerBlock *p = static_cast<TransformerBlock*>(layer);
        lerp(p->gamma1, gamma1, alpha);
        lerp(p->beta1,  beta1,  alpha);
        lerp(p->gamma2, gamma2, alpha);
        lerp(p->beta2,  beta2,  alpha);
        attn.softUpdateTo(&p->attn, alpha);
        ffn_up.softUpdateTo(&p->ffn_up, alpha);
        ffn_down.softUpdateTo(&p->ffn_down, alpha);
        return;
    }

    void write(std::ofstream &file) override
    {
        file << gamma1.toString() << std::endl;
        file << beta1.toString()  << std::endl;
        file << gamma2.toString() << std::endl;
        file << beta2.toString()  << std::endl;
        attn.write(file);
        ffn_up.write(file);
        ffn_down.write(file);
        return;
    }

    void read(std::ifstream &file) override
    {
        std::string s;
        std::getline(file, s); gamma1 = Tensor::fromString(s);
        std::getline(file, s); beta1  = Tensor::fromString(s);
        std::getline(file, s); gamma2 = Tensor::fromString(s);
        std::getline(file, s); beta2  = Tensor::fromString(s);
        attn.read(file);
        ffn_up.read(file);
        ffn_down.read(file);
        return;
    }
};

}
#endif // TRANSFORMER_HPP
