#!/usr/bin/env bash
# install Boost package
sudo apt-get update
sudo apt-get install swig build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
git clone https://github.com/NVIDIA/OpenSeq2Seq
cd OpenSeq2Seq
git checkout ctc-decoders
cd ..
mv OpenSeq2Seq/decoders .
rm -rf OpenSeq2Seq
cd decoders
# patch setup code to support the recent distutils
sed -i 's/, distutils/, distutils\nimport distutils.ccompiler/g' setup.py
./setup.sh
cd ..
