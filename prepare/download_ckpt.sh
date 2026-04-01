#!/bin/bash

rm -rf ckpt
echo -e "Downloading FrameCrafter ckpts"
mkdir ckpt
cd ckpt
gdown --fuzzy https://drive.google.com/file/d/1eaEay33yAfE8IJJdO-avZZCr-eGD4OZK/view

unzip framecrafter.zip

echo -e "Cleaning\n"
rm framecrafter.zip
echo -e "Downloading done!"
cd ..