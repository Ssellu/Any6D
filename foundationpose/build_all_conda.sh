#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Building FoundationPose extensions..."

# PyTorch 설치 확인
python -c "import torch; print(f'PyTorch {torch.__version__} found')"

# Install mycpp
echo "Building mycpp..."
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "mycpp build successful!"
    cp *.so ../
else
    echo "Error: mycpp build failed!"
    exit 1
fi

# Install mycuda (여러 방법 시도)
echo "Building mycuda..."
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so

# 방법 1: --no-use-pep517 + --no-build-isolation
echo "Trying method 1: --no-use-pep517 --no-build-isolation"
python -m pip install -e . --no-use-pep517 --no-build-isolation --verbose && echo "Method 1 successful!" || {
    echo "Method 1 failed, trying method 2: direct setup.py"
    
    # 방법 2: 직접 setup.py 실행
    python setup.py develop --no-deps && echo "Method 2 successful!" || {
        echo "Method 2 failed, trying method 3: manual build"
        
        # 방법 3: 수동 빌드
        python setup.py build_ext --inplace && python setup.py develop --no-deps && echo "Method 3 successful!" || {
            echo "All methods failed!"
            exit 1
        }
    }
}

cd ${PROJ_ROOT}
echo "Build completed!"