mkdir -p ~/src && cd ~/src && git clone --depth 50 https://github.com/ggml-org/llama.cpp.git 2>&1 | tail -2
cd llama.cpp && git log -1 --format='%H %cd %s'
