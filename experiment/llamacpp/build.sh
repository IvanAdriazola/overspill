export PATH=/usr/local/cuda/bin:$PATH
cd ~/src/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc > /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/cmake_config.log 2>&1 || { echo CONFIG FAILED; tail -30 /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/cmake_config.log; exit 1; }
cmake --build build --config Release -j 10 --target llama-server llama-cli llama-bench > /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/build.log 2>&1
rc=$?; echo "BUILD RC=$rc"; tail -5 /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/build.log
ls -la build/bin/
