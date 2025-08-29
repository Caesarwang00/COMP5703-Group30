训练集：70% 样本

临时集：30% 样本 临时集一半作验证集，一半作测试集

最终比例：

训练集：70%

验证集：15%

测试集：15%

python train.py --label\_column GeneExp\_Subtype --discrete\_input true
python test.py

