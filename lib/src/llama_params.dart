/// Params holder like `gpt_params` in `common/common.h`
final class LlamaParams {
  final int? seed;
  final int? nThread;
  final int? nThreadBatch;
  final int? nPredict;
  final int? nCtx;
  final int? nBatch;
  final int? nGpuLayers;
  final int? mainGpu;
  final bool embedding;
  final int numa;

  const LlamaParams({
    this.seed,
    this.nThread,
    this.nThreadBatch,
    this.nPredict,
    this.nCtx,
    this.nBatch,
    this.nGpuLayers,
    this.mainGpu,
    this.embedding = false,
    this.numa = 0,
  });
}
