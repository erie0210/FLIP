import sys
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movielens')  # movielens, bookcrossing, goodreads
parser.add_argument('--backbone', type=str, default='DCNv2')     # DeepFM, AutoInt, DCNv2
parser.add_argument('--llm', type=str, default='tiny-bert')      # tiny-bert, roberta, roberta-large
parser.add_argument('--epochs', type=int, default=30)
# ↓ 추가된 인자들 (기존 코드에서 참조하고 있었으나 정의되어 있지 않던 것들)
parser.add_argument('--init_method', type=str, default='tcp://localhost:23456')
parser.add_argument('--train_url', type=str, default='')

add_args = parser.parse_args()

TARGET_PY_FILE = 'pretrain_MaskCTR_ddp.py'

# Kaggle에서는 단일 GPU로 실행
PREFIX = ['python', TARGET_PY_FILE]

if add_args.llm == 'tiny-bert':
    MIXED_PRECISION = False
    batch_size = 128
elif add_args.llm == 'roberta':
    MIXED_PRECISION = True
    batch_size = 64
elif add_args.llm == 'roberta-large':
    MIXED_PRECISION = True
    batch_size = 16

SAMPLE = False

for EPOCHS in [add_args.epochs]:
    for BS in [batch_size]:
        for DATASET in [add_args.dataset]:
            for TEM in [0.7]:
                for LR in [1e-4]:
                    for USE_MFM in [True]:
                        for USE_MLM in [True]:
                            for BACKBONE in [add_args.backbone]:
                                for USE_ATTENTION in [True]:
                                    subprocess.run(PREFIX + [
                                        f'--backbone={BACKBONE}',
                                        f'--temperature={TEM}',
                                        f'--use_mfm={USE_MFM}',
                                        f'--use_mlm={USE_MLM}',
                                        f'--epochs={EPOCHS}',
                                        f'--lr={LR}',
                                        f'--batch_size={BS}',
                                        f'--dataset={DATASET}',
                                        f'--sample={SAMPLE}',
                                        f'--mixed_precision={MIXED_PRECISION}',
                                        f'--llm={add_args.llm}',
                                        f'--use_attention={USE_ATTENTION}'
                                    ])
