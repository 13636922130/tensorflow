#!/bin/bash
#把训练集中的12500张狗的图像和12500张猫的图像
#分别取出2500张作为测试集

echo "The shell is running..."

for i in `seq 10000 12499`
do
    echo "move dog.$i.jpg"
    mv train/dog.$i.jpg test/
    echo "move cat.$i.jpg"
    mv train/cat.$i.jpg test/
done
