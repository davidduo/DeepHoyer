
echo $(date +%F%n%T)
python3 element.py --reg_type 1 --decay 0.001  --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 1 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 1 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 1 --decay 0.00005 --pretrained
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_1 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_1 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_1 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_1 --sensitivity 0.005
echo $(date +%F%n%T)




echo $(date +%F%n%T)
python3 element.py --reg_type 2 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 2 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 2 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 2 --decay 0.00005 --pretrained
echo $(date +%F%n%T)

python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_2 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_2 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_2 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_2 --sensitivity 0.005
echo $(date +%F%n%T)




echo $(date +%F%n%T)
python3 element.py --reg_type 3 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 3 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 3 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 3 --decay 0.00005 --pretrained
echo $(date +%F%n%T)

python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_3 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_3 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_3 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_3 --sensitivity 0.005
echo $(date +%F%n%T)



echo $(date +%F%n%T)
python3 element.py --reg_type 4 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 4 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 4 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 4 --decay 0.00005 --pretrained
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_4 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_4 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_4 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_4 --sensitivity 0.005
echo $(date +%F%n%T)




echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.001  --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.00005 --pretrained
echo $(date +%F%n%T)

python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_5 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_5 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_5 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_5 --sensitivity 0.005
echo $(date +%F%n%T)




echo $(date +%F%n%T)
python3 element.py --reg_type 6 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 6 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 6 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 6 --decay 0.00005 --pretrained
echo $(date +%F%n%T)

python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_6 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_6 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_6 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_6 --sensitivity 0.005
echo $(date +%F%n%T)




echo $(date +%F%n%T)
python3 element.py --reg_type 7 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 7 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 7 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 7 --decay 0.00005 --pretrained
echo $(date +%F%n%T)

python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_7 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_7 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_7 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_7 --sensitivity 0.005
echo $(date +%F%n%T)




echo $(date +%F%n%T)
python3 element.py --reg_type 8 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 8 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 8 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 8 --decay 0.00005 --pretrained
echo $(date +%F%n%T)


python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.001_8 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0005_8 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.0001_8 --sensitivity 0.005
echo $(date +%F%n%T)
python3 prun_tune_T.py --model /home/david/PycharmProjects/DeepHoyer/mnist/MLP/saves/elt_0.00005_8 --sensitivity 0.005
echo $(date +%F%n%T)

