#ifndef ATTENTION_HPP
#define ATTENTION_HPP
#include <memory>
#include <fstream>
#include "tensor.hpp"
#include "ilayer.h"
#include "activate.h"
#include "optimize.h"

namespace RL {

class PositionalEncoder : public iLayer
{
public:
    int inputDim;
    int pos;
    Tensor pe;
public:
    PositionalEncoder(){}
    explicit PositionalEncoder(int inputDim_, bool withGrad_)
        :inputDim(inputDim_)
    {
        o = Tensor(inputDim, 1);
        e = Tensor(inputDim, 1);
    }

    Tensor& forward(const Tensor& x, bool inference=false) override
    {
        float d = x.totalSize;
        for (std::size_t i = 0; i < x.totalSize; i++) {
            if (i%2 == 0) {
                pe[i] = std::sin(float(pos)/std::pow(10000, float(i)/d));
            } else {
                pe[i] = std::cos(float(pos)/std::pow(10000, float(i - 1)/d));
            }
            o[i] = x[i] + pe[i];
        }
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {

    }
    void SGD(float lr) override
    {

    }
    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {

    }
    void Adam(float lr, float alpha, float beta,
                      float alpha_, float beta_,
                      float decay, bool clipGrad) override
    {

    }
    void clamp(float c0, float cn) override
    {

    }
    void copyTo(iLayer* layer) override
    {

    }
    void softUpdateTo(iLayer* layer, float alpha) override
    {

    }
};


class ScaledDotProduct : public iLayer
{
public:
    class ScaledDotProductGrad
    {
    public:
        Tensor wq;
        Tensor wk;
        Tensor wv;
    public:
        ScaledDotProductGrad(){}
        void zero()
        {
            wq.zero();
            wk.zero();
            wv.zero();
            return;
        }
    };
public:
    int inputDim;
    int outputDim;
    float d;
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor z;
    ScaledDotProductGrad g;
    ScaledDotProductGrad gv;
    ScaledDotProductGrad gm;
public:
    ScaledDotProduct(){}

    explicit ScaledDotProduct(int inputDim_, int outputDim_, bool withGrad_)
        :inputDim(inputDim_),outputDim(outputDim_)
    {
        type = LAYER_SCALEDDOTPRODUCT;
        wq = Tensor(outputDim, inputDim);
        wk = Tensor(outputDim, inputDim);
        wv = Tensor(outputDim, inputDim);
        Random::uniform(wq, -1, 1);
        Random::uniform(wk, -1, 1);
        Random::uniform(wv, -1, 1);
        q = Tensor(outputDim, 1);
        k = Tensor(outputDim, 1);
        v = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        z = Tensor(outputDim, outputDim);
        if (withGrad_) {
            g.wq = Tensor(outputDim, inputDim);
            g.wk = Tensor(outputDim, inputDim);
            g.wv = Tensor(outputDim, inputDim);
            gv.wq = Tensor(outputDim, inputDim);
            gv.wk = Tensor(outputDim, inputDim);
            gv.wv = Tensor(outputDim, inputDim);
            gm.wq = Tensor(outputDim, inputDim);
            gm.wk = Tensor(outputDim, inputDim);
            gm.wv = Tensor(outputDim, inputDim);
        }
        d = std::sqrt(outputDim);
    }

    static std::shared_ptr<ScaledDotProduct> _(int inputDim, int outputDim, bool withGrad)
    {
        return std::make_shared<ScaledDotProduct>(inputDim, outputDim, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        Tensor::MM::ikkj(q, wq, x);
        Tensor::MM::ikkj(k, wk, x);
        Tensor::MM::ikkj(v, wv, x);
        Tensor::MM::ikjk(z, q, k);
        z /= d;
        softmax(z);
        Tensor::MM::ikkj(o, z, v);
        return o;
    }

    /*
        Reference implementation: backward using explicit O(N⁴) softmax Jacobian matrix.

        Constructs the full N²×N² Jacobian J(i,j) = z[i]·(δ_ij - z[j]),
        then computes vec(dz) = J^T · vec(dz_hat) via kikj.

        This is mathematically correct but O(N⁴) time and O(N⁴) memory.
        Use backward() instead for O(N²) performance.
    */
    void backward_jacobian(Tensor &ei)
    {
        Tensor dv(outputDim, 1);
        Tensor::MM::kikj(dv, z, e);

        Tensor dz_hat(outputDim, outputDim);
        Tensor::MM::ikjk(dz_hat, e, v);

        /* Explicit N²×N² Jacobian matrix construction (O(N⁴) memory + time) */
        Tensor J = Softmax::jacobian(z);           // N² × N² matrix!
        Tensor dz_hat_vec = dz_hat;
        dz_hat_vec.reshape(outputDim*outputDim, 1);
        Tensor dz_vec(outputDim*outputDim, 1);
        Tensor::MM::kikj(dz_vec, J, dz_hat_vec);   // J^T · vec(dz_hat)
        dz_vec.reshape(outputDim, outputDim);

        Tensor dq(outputDim, 1);
        Tensor::MM::ikkj(dq, dz_vec, k);
        dq /= d;

        Tensor dk(outputDim, 1);
        Tensor dz_t = dz_vec.tr();
        Tensor::MM::ikkj(dk, dz_t, q);
        dk /= d;

        Tensor::MM::kikj(ei, wq, dq);
        Tensor::MM::kikj(ei, wk, dk);
        Tensor::MM::kikj(ei, wv, dv);
        return;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /*
            Correct backward through ScaledDotProduct:
            o = ẑ · v, ẑ = softmax(q·k^T / d)

            Given e = ∂L/∂o (N×1):
            1. ∂L/∂v = ẑ^T · e                          (N×1)
            2. ∂L/∂ẑ = e · v^T                          (N×N)
            3. vec(∂L/∂z) = J^T · vec(∂L/∂ẑ)           (O(N²) via jacobian_transpose_mul)
            4. ∂L/∂q = (∂L/∂z) · k / d                  (N×1)
            5. ∂L/∂k = (∂L/∂z)^T · q / d                (N×1)
            6. ∂L/∂x = Wq^T·∂L/∂q + Wk^T·∂L/∂k + Wv^T·∂L/∂v
            7. g.wq += ∂L/∂q · x^T, etc.
        */

        /* Step 1: ∂L/∂v = z^T · e  (kikj: result(i) = Σ_k z(k,i)·e(k)) */
        Tensor dv(outputDim, 1);
        Tensor::MM::kikj(dv, z, e);

        /* Step 2: ∂L/∂ẑ = e · v^T  (ikjk: dz_hat(i,j) = e(i)·v(j)) */
        Tensor dz_hat(outputDim, outputDim);
        Tensor::MM::ikjk(dz_hat, e, v);

        /* Step 3: ∂L/∂z = J^T · ∂L/∂ẑ  (O(N²) softmax Jacobian-vector product) */
        Tensor dz(outputDim, outputDim);
        Softmax::jacobian_transpose_mul(z, dz_hat, dz);

        /* Step 4: ∂L/∂q = dz · k / d  */
        Tensor dq(outputDim, 1);
        Tensor::MM::ikkj(dq, dz, k);
        dq /= d;

        /* ∂L/∂k = dz^T · q / d */
        Tensor dk(outputDim, 1);
        Tensor dz_t = dz.tr();
        Tensor::MM::ikkj(dk, dz_t, q);
        dk /= d;

        /* Step 5: ∂L/∂x = Wq^T·dq + Wk^T·dk + Wv^T·dv
           NOTE: DO NOT zero ei here! May be called from multi-head attention
           which accumulates all heads' gradients. */
        Tensor::MM::kikj(ei, wq, dq);
        Tensor::MM::kikj(ei, wk, dk);
        Tensor::MM::kikj(ei, wv, dv);

        /* Step 6: Parameter gradients */
        Tensor::MM::ikjk(g.wq, dq, x);    // g.wq += dq · x^T
        Tensor::MM::ikjk(g.wk, dk, x);    // g.wk += dk · x^T
        Tensor::MM::ikjk(g.wv, dv, x);    // g.wv += dv · x^T

        /* Clear cached state */
        q.zero();
        k.zero();
        v.zero();
        z.zero();
        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        Optimize::SGD(wq, g.wq, lr);
        Optimize::SGD(wk, g.wk, lr);
        Optimize::SGD(wv, g.wv, lr);
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wq, gv.wq, g.wq, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wk, gv.wk, g.wk, lr, rho, decay, clipGrad);
        Optimize::RMSProp(wv, gv.wv, g.wv, lr, rho, decay, clipGrad);
        g.zero();
        return;
    }

     void Adam(float lr, float alpha, float beta,
               float alpha_, float beta_,
               float decay, bool clipGrad) override
    {
        Optimize::Adam(wq, gv.wq, gm.wq, g.wq,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(wk, gv.wk, gm.wk, g.wk,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(wv, gv.wv, gm.wv, g.wv,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(wq, c0, cn);
         Optimize::clamp(wk, c0, cn);
         Optimize::clamp(wv, c0, cn);
         return;
     }

     virtual void copyTo(iLayer* layer) override
     {
         ScaledDotProduct *pLayer = static_cast<ScaledDotProduct*>(layer);
         pLayer->wq = wq;
         pLayer->wk = wk;
         pLayer->wv = wv;
         return;
     }
     virtual void softUpdateTo(iLayer* layer, float alpha) override
     {
         ScaledDotProduct *pLayer = static_cast<ScaledDotProduct*>(layer);
         lerp(pLayer->wq, wq, alpha);
         lerp(pLayer->wk, wk, alpha);
         lerp(pLayer->wv, wv, alpha);
         return;
     }

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<wq.toString()<<std::endl;
         file<<wk.toString()<<std::endl;
         file<<wv.toString()<<std::endl;
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string wqs;
         std::getline(file, wqs);
         wq = Tensor::fromString(wqs);
         std::string wks;
         std::getline(file, wks);
         wk = Tensor::fromString(wks);
         std::string wvs;
         std::getline(file, wvs);
         wv = Tensor::fromString(wvs);
         return;
     }
};

template<int N>
class Attention : public iLayer
{
public:
    class AttentionGrad
    {
    public:
        Tensor w1;
        Tensor w2;
        Tensor b;
    public:
        AttentionGrad(){}
        void zero()
        {
            w1.zero();
            w2.zero();
            b.zero();
        }
    };
public:
    int inputDim;
    int unitDim;
    int outputDim;
    Tensor w1;
    Tensor w2;
    Tensor b;
    Tensor a;
    ScaledDotProduct dotProduct[N];
    AttentionGrad g;
    AttentionGrad v;
    AttentionGrad m;
public:
    Attention(){}
    explicit Attention(int inputDim_, int unitDim_, bool withGrad)
        :inputDim(inputDim_),unitDim(unitDim_)
    {
        type = LAYER_ATTENTION;
        outputDim = unitDim*N;
        w1 = Tensor(outputDim, outputDim);
        w2 = Tensor(outputDim, inputDim);
        b = Tensor(outputDim, 1);
        Random::uniform(w1, -1, 1);
        Random::uniform(w2, -1, 1);
        Random::uniform(b, -1, 1);
        for (int i = 0; i < N; i++) {
            dotProduct[i] = ScaledDotProduct(inputDim, unitDim, withGrad);
        }
        a = Tensor(outputDim, 1);
        o = Tensor(outputDim, 1);
        e = Tensor(outputDim, 1);
        if (withGrad) {
            g.w1 = Tensor(outputDim, outputDim);
            v.w1 = Tensor(outputDim, outputDim);
            m.w1 = Tensor(outputDim, outputDim);
            g.w2 = Tensor(outputDim, inputDim);
            v.w2 = Tensor(outputDim, inputDim);
            m.w2 = Tensor(outputDim, inputDim);
            g.b = Tensor(outputDim, 1);
            v.b = Tensor(outputDim, 1);
            m.b = Tensor(outputDim, 1);
        }
    }
    static std::shared_ptr<Attention> _(int inputDim, int outputDim, bool withGrad)
    {
        return std::make_shared<Attention>(inputDim, outputDim, withGrad);
    }
    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        for (int i = 0; i < N; i++) {
            Tensor &out = dotProduct[i].forward(x, inference);
            a.embedding({i*unitDim, 0}, out);
        }
        Tensor::MM::ikkj(o, w1, a);
        Tensor::MM::ikkj(o, w2, x);
        o += b;
        for (std::size_t i = 0; i < o.totalSize; i++) {
            o[i] = Tanh::f(o[i]);
        }
        return o;
    }

    void backward(const Tensor &x, Tensor &ei) override
    {
        /*
            Attention backward:
            o = tanh(z), z = w1·a + w2·x + b
            a = [head₁(x); head₂(x); ...; headₙ(x)]

            Given e = ∂L/∂o (N×1):
            ∂L/∂z = e ⊙ tanh'(o)             → dy
            ∂L/∂x = w2^T · dy                (residual path)
            ∂L/∂a = w1^T · dy                (attention path)
                  = [∂L/∂head₁; ...; ∂L/∂headₙ]
            Each ∂L/∂head_i backpropagates through ScaledDotProduct to ∂L/∂x
        */

        /* ∂L/∂z = e ⊙ tanh'(o) */
        Tensor dy(outputDim, 1);
        for (std::size_t i = 0; i < outputDim; i++) {
            dy[i] = Tanh::df(o[i])*e[i];
        }
        /* w2 residual path: ∂L/∂x += w2^T · dy  */
        Tensor::MM::kikj(ei, w2, dy);

        /* w1 attention path through heads:
           ∂L/∂a = w1^T · dy
           then backprop through each head */
        Tensor da(outputDim, 1);
        Tensor::MM::kikj(da, w1, dy);
        for (int i = 0; i < N; i++) {
            /*
                Broadcast the correct gradient to each ScaledDotProduct head.
                o = tanh(w1·a + w2·x + b), a = [head₁...headₙ]
                Given e = ∂L/∂o:
                head_i.e = ∂L/∂(head_i_output)
                = block_of(w1^T · (e ⊙ tanh'(o)), i*unitDim, unitDim)
            */
            dotProduct[i].e = da.block({i*unitDim, 0}, {unitDim, 1});
            dotProduct[i].backward(x, ei);    /* accumulates ∂L/∂x_head into ei */
        }

        /*
            ∂L/∂z = e ⊙ tanh'(o)                  → dy
            g.w1 += dy · a^T
            g.w2 += dy · x^T
            g.b  += dy
            For each head: da_i = block_of(w1^T·dy), gradient(da_i)
        */
        Tensor::MM::ikjk(g.w1, dy, a);
        Tensor::MM::ikjk(g.w2, dy, x);
        g.b += dy;
        o.zero();
        e.zero();
        return;
    }

    void SGD(float learningRate) override
    {
        Optimize::SGD(w1, g.w1, learningRate);
        Optimize::SGD(w2, g.w2, learningRate);
        Optimize::SGD(b, g.b, learningRate);
        for (int i = 0; i < N; i++) {
            dotProduct[i].SGD(learningRate);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(w1, v.w1, g.w1, lr, rho, decay, clipGrad);
        Optimize::RMSProp(w2, v.w2, g.w2, lr, rho, decay, clipGrad);
        Optimize::RMSProp(b, v.b, g.b, lr, rho, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            dotProduct[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(w1, v.w1, m.w1, g.w1,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(w2, v.w2, m.w2, g.w2,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        Optimize::Adam(b, v.b, m.b, g.b,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < N; i++) {
            dotProduct[i].Adam(lr, alpha, beta,
                               alpha_, beta_,
                               decay, clipGrad);
        }
        g.zero();
        return;
    }

     void clamp(float c0, float cn) override
     {
         Optimize::clamp(w1, c0, cn);
         Optimize::clamp(w2, c0, cn);
         Optimize::clamp(b, c0, cn);
         for (int i = 0; i < N; i++) {
             dotProduct[i].clamp(c0, cn);
         }
         return;
     }

     void copyTo(iLayer* layer) override
     {
         Attention *pLayer = static_cast<Attention*>(layer);
         pLayer->w1 = w1;
         pLayer->w2 = w2;
         pLayer->b = b;
         for (int i = 0; i < N; i++) {
            dotProduct[i].copyTo(&pLayer->dotProduct[i]);
         }
         return;
     }

     void softUpdateTo(iLayer* layer, float alpha) override
     {
         Attention *pLayer = static_cast<Attention*>(layer);
         lerp(pLayer->w1, w1, alpha);
         lerp(pLayer->w2, w2, alpha);
         lerp(pLayer->b, b, alpha);
         for (int i = 0; i < N; i++) {
             dotProduct[i].softUpdateTo(&pLayer->dotProduct[i], alpha);
         }
         return;
     }

     virtual void write(std::ofstream &file) override
     {
         /* w */
         file<<w1.toString()<<std::endl;
         file<<w2.toString()<<std::endl;
         /* b */
         file<<b.toString()<<std::endl;
         for (int i = 0; i < N; i++) {
             dotProduct[i].write(file);
         }
         return;
     }

     virtual void read(std::ifstream &file) override
     {
         /* w */
         std::string w1s;
         std::getline(file, w1s);
         w1 = Tensor::fromString(w1s);
         std::string w2s;
         std::getline(file, w2s);
         w2 = Tensor::fromString(w2s);
         std::string bs;
         std::getline(file, bs);
         b = Tensor::fromString(bs);

         for (int i = 0; i < N; i++) {
             dotProduct[i].read(file);
         }
         return;
     }
};

/*
    Standard Multi-Head Attention (Transformer MHA)

    Architecture:
        For each head i (0..h-1) with head_dim = d_k = modelDim / numHeads:
            q_i = Wq_i · x   (d_k × 1)
            k_i = Wk_i · x   (d_k × 1)
            v_i = Wv_i · x   (d_k × 1)
            attn_i = softmax(q_i · k_i^T / √d_k)  (d_k × d_k)
            head_i = attn_i · v_i  (d_k × 1)

        all_heads = [head₀; head₁; ...; head_{h-1}]  (modelDim × 1)
        o = Wo · all_heads  (modelDim × 1)

    Key differences from Attention<N> class:
        - No tanh activation
        - No residual/W2 path
        - Standard output projection Wo
        - Cleaner multi-head gradient accumulation (broadcast + backward)

    Forward:
        output = Wo · concat(head₁(x), ..., headₕ(x))

    Backward (given e = ∂L/∂o, modelDim×1):
        1. ∂L/∂all_heads = Wo^T · e  (modelDim×1)
        2. For each head i: ∂L/∂head_i = block(∂L/∂all_heads, i·d_k, d_k)
        3. Each head_i.backward accumulates ∂L/∂x into ei

    Gradient:
        1. g.Wo += e · (all_heads)^T  (ikjk)
        2. Each head_i.gradient(x, y) computes its Wq/Wk/Wv gradients

    Integration with Net::backward:
        Uses LAYER_MHA type. The same broadcast + backward pattern works.
*/
template<int NumHeads>
class MultiHeadAttention : public iLayer
{
public:
    class MHAGrad
    {
    public:
        Tensor wo;
    public:
        MHAGrad(){}
        void zero() { wo.zero(); }
    };
public:
    int inputDim;
    int d_k;           // head dimension = d_model / NumHeads
    int d_model;       // total model dimension = NumHeads * d_k
    Tensor wo;         // output projection (d_model × d_model)
    Tensor a;          // concatenated head outputs (d_model × 1)
    ScaledDotProduct heads[NumHeads];
    MHAGrad g;
    MHAGrad v;
    MHAGrad m;
public:
    MultiHeadAttention(){}
    explicit MultiHeadAttention(int inputDim_, int d_model_, bool withGrad)
        :inputDim(inputDim_), d_model(d_model_)
    {
        type = LAYER_MHA;
        d_k = d_model / NumHeads;
        /* Ensure d_model is divisible by NumHeads */
        if (d_k * NumHeads != d_model) {
            d_k = d_model / NumHeads + 1;
            d_model = d_k * NumHeads;
        }

        wo = Tensor(d_model, d_model);
        Random::uniform(wo, -1, 1);
        for (int i = 0; i < NumHeads; i++) {
            heads[i] = ScaledDotProduct(inputDim, d_k, withGrad);
        }
        a = Tensor(d_model, 1);
        o = Tensor(d_model, 1);
        e = Tensor(d_model, 1);
        if (withGrad) {
            g.wo = Tensor(d_model, d_model);
            v.wo = Tensor(d_model, d_model);
            m.wo = Tensor(d_model, d_model);
        }
    }

    static std::shared_ptr<MultiHeadAttention> _(int inputDim, int d_model, bool withGrad)
    {
        return std::make_shared<MultiHeadAttention>(inputDim, d_model, withGrad);
    }

    Tensor& forward(const RL::Tensor &x, bool inference=false) override
    {
        /*
            Forward: o = Wo · concat(head₀(x), ..., head_{h-1}(x))
            Each head_i(x) = softmax(Wq_i·x · (Wk_i·x)^T / √d_k) · Wv_i·x  (d_k × 1)
        */
        for (int i = 0; i < NumHeads; i++) {
            Tensor &head_out = heads[i].forward(x, inference);
            a.embedding({i*d_k, 0}, head_out);
        }
        /* o = Wo · a */
        Tensor::MM::ikkj(o, wo, a);
        return o;
    }

    void backward(const Tensor& x, Tensor &ei) override
    {
        /*
            ∂L/∂all_heads = Wo^T · e
            For each head i:
                ∂L/∂head_i = block(Wo^T·e, i*d_k, d_k)
                head_i.backward(ei)  (accumulates ∂L/∂x_head_i into ei)
        */
        /* ∂L/∂all_heads = Wo^T · e */
        Tensor da(d_model, 1);
        Tensor::MM::kikj(da, wo, e);
        ei += da;
        /* Backprop through each head into ei */
        for (int i = 0; i < NumHeads; i++) {
            /*
                Distribute the output gradient to each head for subsequent gradient()
                ∂L/∂head_i = block(Wo^T · e, i*d_k, d_k)
            */
            heads[i].e = da.block({i*d_k, 0}, {d_k, 1});
            heads[i].backward(x, ei);   /* accumulates ∂L/∂x_head into ei */
        }

        /*
            g.Wo += e · a^T
            For each head i: head_i.gradient(x, y)
        */
        Tensor::MM::ikjk(g.wo, e, a);

        o.zero();
        e.zero();
        return;
    }

    void SGD(float lr) override
    {
        Optimize::SGD(wo, g.wo, lr);
        for (int i = 0; i < NumHeads; i++) {
            heads[i].SGD(lr);
        }
        g.zero();
        return;
    }

    void RMSProp(float lr, float rho, float decay, bool clipGrad) override
    {
        Optimize::RMSProp(wo, v.wo, g.wo, lr, rho, decay, clipGrad);
        for (int i = 0; i < NumHeads; i++) {
            heads[i].RMSProp(lr, rho, decay, clipGrad);
        }
        g.zero();
        return;
    }

    void Adam(float lr, float alpha, float beta,
              float alpha_, float beta_,
              float decay, bool clipGrad) override
    {
        Optimize::Adam(wo, v.wo, m.wo, g.wo,
                       alpha_, beta_, lr,
                       alpha, beta, decay, clipGrad);
        for (int i = 0; i < NumHeads; i++) {
            heads[i].Adam(lr, alpha, beta,
                          alpha_, beta_,
                          decay, clipGrad);
        }
        g.zero();
        return;
    }

    void clamp(float c0, float cn) override
    {
        Optimize::clamp(wo, c0, cn);
        for (int i = 0; i < NumHeads; i++) {
            heads[i].clamp(c0, cn);
        }
        return;
    }

    void copyTo(iLayer* layer) override
    {
        MultiHeadAttention *pLayer = static_cast<MultiHeadAttention*>(layer);
        pLayer->wo = wo;
        for (int i = 0; i < NumHeads; i++) {
            heads[i].copyTo(&pLayer->heads[i]);
        }
        return;
    }

    void softUpdateTo(iLayer* layer, float alpha) override
    {
        MultiHeadAttention *pLayer = static_cast<MultiHeadAttention*>(layer);
        lerp(pLayer->wo, wo, alpha);
        for (int i = 0; i < NumHeads; i++) {
            heads[i].softUpdateTo(&pLayer->heads[i], alpha);
        }
        return;
    }

    void write(std::ofstream &file) override
    {
        file << wo.toString() << std::endl;
        for (int i = 0; i < NumHeads; i++) {
            heads[i].write(file);
        }
        return;
    }

    void read(std::ifstream &file) override
    {
        std::string wos;
        std::getline(file, wos);
        wo = Tensor::fromString(wos);
        for (int i = 0; i < NumHeads; i++) {
            heads[i].read(file);
        }
        return;
    }
};

}
#endif // ATTENTION_HPP
