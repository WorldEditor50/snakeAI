#ifndef MOE_HPP
#define MOE_HPP
#include <memory>
#include <iostream>
#include "util.hpp"
#include "optimize.h"
#include "activate.h"
#include "ilayer.h"
#include "layer.h"
#include "transformer.hpp"

namespace RL {

/*
    MOE<NumExperts, NumHeads> — Mixture of Experts with TransformerBlock experts

    Architecture (Pre-LN TransformerBlock as each expert):

        Input x (d_model × 1):

        1. Gating: gate_logits = Wg · x + b        (NumExperts × 1)
                   gate       = softmax(gate_logits)

        2. For each expert i (TransformerBlock<NumHeads>):
               expert_output_i = TransformerBlock_i(x)   (d_model × 1)

        3. Weighted output: o = Σ_i gate[i] · expert_output_i   (d_model × 1)

        Each expert is a full Pre-LN TransformerBlock:
            x_norm1 = LayerNorm(x)
            attn_out = MultiHeadAttention<NumHeads>(x_norm1)
            x_res1 = x + attn_out
            x_norm2 = LayerNorm(x_res1)
            ffn_out = Linear(GELU(Linear(x_norm2)))
            expert_output = x_res1 + ffn_out

    Gradient flow:
        e = dL/do  (d_model × 1)

        dL/d_expert_output[i] = gate[i] · e          (d_model × 1)
        dL/d_gate[i]          = e · expert_output[i]  (scalar)

        dL/dx = Σ_i TransformerBlock_i.backward(dL/d_expert_output[i])
                + Wg^T · (J_softmax^T · dL/d_gate)

    Type registration: LAYER_MOE
*/
template<int NumExperts, int NumHeads = 4>
class MOE : public iLayer
{
public:
    struct MOEGrad {
        Tensor wg;
        Tensor b;
        MOEGrad() {}
        void zero() { wg.zero(); b.zero(); }
    };

    int d_model;             // model dimension (shared across all experts)
    int d_ff_;               // FFN hidden dimension (defaults to 4*d_model)

    /* Gating network */
    Tensor wg;               // gating weights (NumExperts × d_model)
    Tensor b;                // gating bias     (NumExperts × 1)
    //Tensor gate_logits;      // pre-softmax gating logits (NumExperts × 1)
    Tensor gate;             // post-softmax gating weights (NumExperts × 1)

    /* Experts — each is a full TransformerBlock */
    TransformerBlock<NumHeads> experts[NumExperts];

    /* Cached intermediate values for backward/gradient */
    Tensor expert_out[NumExperts];  // each expert's output (d_model × 1)

    /* Gradients */
    MOEGrad g, v, m;

public:
    MOE() {}
    virtual ~MOE() {}

    static std::shared_ptr<MOE> _(
        int d_model_, bool withGrad)
    {
        return std::make_shared<MOE>(d_model_, withGrad);
    }

    explicit MOE(int d_model_, bool withGrad)
        : d_model(d_model_)
    {
        type = LAYER_MOE;
        d_ff_ = 4 * d_model;

        /* Gating network parameters */
        wg = Tensor(NumExperts, d_model);
        b  = Tensor(NumExperts, 1);
        Random::uniform(wg, -0.1f, 0.1f);
        Random::uniform(b,  -0.1f, 0.1f);
        gate        = Tensor(NumExperts, 1);

        /* Experts — each a full TransformerBlock */
        for (int i = 0; i < NumExperts; i++) {
            experts[i] = TransformerBlock<NumHeads>(d_model, withGrad);
        }

        /* Output and error */
        o = Tensor(d_model, 1);
        e = Tensor(d_model, 1);

        /* Cached intermediates */
        //x_orig = Tensor(d_model, 1);
        for (int i = 0; i < NumExperts; i++) {
            expert_out[i] = Tensor(d_model, 1);
        }

        /* Gradient tensors */
        if (withGrad) {
            g.wg = Tensor(NumExperts, d_model);
            g.b  = Tensor(NumExperts, 1);
            v.wg = Tensor(NumExperts, d_model);
            v.b  = Tensor(NumExperts, 1);
            m.wg = Tensor(NumExperts, d_model);
            m.b  = Tensor(NumExperts, 1);
        }
    }

    /* ==================== Forward ==================== */

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        /* Save original input for backward */
        //x_orig = x;

        /* Step 1: Compute gating distribution */
        /* gate_logits = Wg · x + b */
        Tensor::MM::ikkj(gate, wg, x);
        gate += b;
        softmax(gate);        // in-place softmax

        /* Step 2: Compute each expert's output and weighted combination */
        o.zero();
        for (int i = 0; i < NumExperts; i++) {
            /* Forward through TransformerBlock expert */
            Tensor &exp_out = experts[i].forward(x, inference);
            expert_out[i] = exp_out;  // cache

            /* o += gate[i] * expert_output_i */
            float gi = gate[i];
            if (gi > 1e-8f) {  // skip negligible contributions
                for (int j = 0; j < d_model; j++) {
                    o[j] += gi * expert_out[i][j];
                }
            }
        }

        return o;
    }

    /* ==================== Backward ==================== */
    /*
        Gradient flow:

        o = Σ_i gate[i] · expert_out[i]

        e = dL/do  (d_model × 1)

        ∂L/∂expert_out[i] = gate[i] · e          (d_model × 1)
        ∂L/∂gate[i]       = e · expert_out[i]     (scalar, dot product)

        dL/dx = Σ_i expert_i.backward(∂L/∂expert_out[i])
                + Wg^T · (J^T · ∂L/∂gate_logit)

        where J = ∂softmax/∂gate_logit (softmax Jacobian, NumExperts×NumExperts)
    */
    void backward(const Tensor& x, Tensor &ei) override
    {
        /* ---- Gating path ---- */
        /* ∂L/∂gate[i] = e · expert_out[i]  (scalar dot product) */
        Tensor d_gate(NumExperts, 1);
        for (int i = 0; i < NumExperts; i++) {
            float dgi = 0;
            for (int j = 0; j < d_model; j++) {
                dgi += e[j] * expert_out[i][j];
            }
            d_gate[i] = dgi;
        }

        /* ∂L/∂gate_logit = J^T · ∂L/∂gate  (softmax Jacobian-vector product) */
        Tensor d_gate_logit(NumExperts, 1);
        Softmax::jacobian_transpose_mul(gate, d_gate, d_gate_logit);

        /* ∂L/∂x_gate = Wg^T · ∂L/∂gate_logit (accumulate into ei) */
        Tensor::MM::kikj(ei, wg, d_gate_logit);

        /* ---- Expert path ---- */
        for (int i = 0; i < NumExperts; i++) {
            float gi = gate[i];
            if (gi < 1e-8f) continue;

            /* ∂L/∂expert_out[i] = gate[i] · e */
            for (int j = 0; j < d_model; j++) {
                experts[i].e[j] = gi * e[j];
            }

            /* Backward through expert, accumulates ∂L/∂x_expert into ei */
            experts[i].backward(x, ei);
        }

        /* ---- Gating parameter gradients ---- */
        Tensor::MM::ikjk(g.wg, d_gate_logit, x);   // g.wg += d_gate_logit · x^T
        g.b += d_gate_logit;                        // g.b  += d_gate_logit

        /* Reset output and error */
        o.zero();
        e.zero();
        return;
    }

    /* ==================== Optimizers ==================== */

    void SGD(float lr) override
    {
        Optimize::SGD(wg, g.wg, lr);
        Optimize::SGD(b,  g.b,  lr);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].SGD(lr);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wg, v.wg, g.wg, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b,  v.b,  g.b,  lr, rho, decay, clipGrad);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(wg, v.wg, m.wg, g.wg,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b,  v.b,  m.b,  g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].Adam(lr, alpha, beta,
                            alpha_, beta_,
                            decay, clipGrad);
        }
        g.zero();
        return;
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(wg, c0, cn);
        Optimize::clamp(b,  c0, cn);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].clamp(c0, cn);
        }
        return;
    }

    void copyTo(iLayer* layer) override
    {
        MOE *p = static_cast<MOE*>(layer);
        p->wg = wg;
        p->b  = b;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].copyTo(&p->experts[i]);
        }
        return;
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        MOE *p = static_cast<MOE*>(layer);
        lerp(p->wg, wg, alpha);
        lerp(p->b,  b,  alpha);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].softUpdateTo(&p->experts[i], alpha);
        }
        return;
    }

    void write(std::ofstream &file) override
    {
        file << wg.toString() << std::endl;
        file << b.toString()  << std::endl;
        for (int i = 0; i < NumExperts; i++) {
            experts[i].write(file);
        }
        return;
    }

    void read(std::ifstream &file) override
    {
        std::string s;
        std::getline(file, s); wg = Tensor::fromString(s);
        std::getline(file, s); b  = Tensor::fromString(s);
        for (int i = 0; i < NumExperts; i++) {
            experts[i].read(file);
        }
        return;
    }
};

}
#endif // MOE_HPP
