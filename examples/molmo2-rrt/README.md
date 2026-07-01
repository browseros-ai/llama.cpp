# Molmo2 RRT runner

Standalone binary:

```text
build/bin/llama-molmo2-rrt-run
```

Build from the llama.cpp worktree root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-molmo2-rrt-run -j8
```

After the first build, rebuild with:

```bash
cmake --build build --target llama-molmo2-rrt-run -j8
```

Required GGUFs:

```text
molmo2-qwen06-rrt-step500-text-f16.gguf
molmo2-qwen06-rrt-step500-vision-f16.gguf
```

Run with Metal/MPS:

```bash
./build/bin/llama-molmo2-rrt-run \
  --model /path/to/molmo2-qwen06-rrt-step500-text-f16.gguf \
  --vision /path/to/molmo2-qwen06-rrt-step500-vision-f16.gguf \
  --image /path/to/image.png \
  --prompt "Click on the target button." \
  --vision-backend mps \
  --threads 8 \
  -ngl 99 \
  -n 64 \
  --no-fa
```

CPU fallback:

```bash
./build/bin/llama-molmo2-rrt-run \
  --model /path/to/molmo2-qwen06-rrt-step500-text-f16.gguf \
  --vision /path/to/molmo2-qwen06-rrt-step500-vision-f16.gguf \
  --image /path/to/image.png \
  --prompt "Click on the target button." \
  --vision-backend cpu \
  --threads 8 \
  -ngl 0 \
  -n 64 \
  --no-fa
```

Notes:

- This is a standalone runner, not `llama-cli`, `llama-server`, or `llama-mtmd-cli`.
- It requires both GGUFs: text/RRT and SigLIP2 vision sidecar.
- Output is point-style XML, for example `<points coords="1 1 195 069">...</points>`.
- Coordinates are normalized to `0..1000`; convert to pixels with `x / 1000 * image_width`, `y / 1000 * image_height`.
