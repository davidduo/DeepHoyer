echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.001 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.0005 --pretrained
echo $(date +%F%n%T)
python3 element.py --reg_type 5 --decay 0.0001 --pretrained
echo $(date +%F%n%T)
