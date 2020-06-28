import os

# Train base model
cmd = 'python3 element.py --reg  0 --decay 0.0  --epoch 250'
os.system(cmd)

# Training for sparsity
for reg_type in range(1, 7, 1):
    for decay in [0.001]:
        for sensitivity in [0.01, 0.001]:
            cmd = 'python3 element.py --reg_type {} --epochs 100 --decay 0.001  --pretrained'.format(reg_type)
            os.system(cmd)

            with open('./model_name.txt', 'r') as fp:
                model_name = fp.readline().strip()[0:-4]
            os.system('rm {}'.format('./model_name.txt'))

            cmd = 'python3 prun_tune_T.py --model {} --epochs 100 --sensitivity {}'.format(model_name, sensitivity)
            print(cmd)
            os.system(cmd)

