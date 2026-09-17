// v620-pp3: standalone torch extension exposing opengfx1030/vllm-rdna (rdna_extras) fused
// W4A16 MoE HIP kernel for gfx1030 to the leapdragon tree, under its own op namespace.
#include <torch/all.h>
#include <torch/library.h>

void moe_gptq_gemm_rdna2(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

TORCH_LIBRARY(_v620_rdna2, m) {
  m.def(
      "moe_gptq_gemm_rdna2(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor(a) b_scales, Tensor b_qzeros, Tensor(a) topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  m.impl("moe_gptq_gemm_rdna2", torch::kCUDA, &moe_gptq_gemm_rdna2);
}
