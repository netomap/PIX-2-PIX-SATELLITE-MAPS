import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from random import choice, random, randint, shuffle
from models import Discriminator, Generator
from dataset import Dataset_Maps, Image_Transformer
from argparse import ArgumentParser
from time import time
from datetime import datetime
from tqdm import tqdm
import pathlib
import warnings
warnings.filterwarnings('ignore')

def debug(frase):
    frase = f'[{datetime.strftime(datetime.now(), "%H:%M:%S")}]: {frase}'
    with open(LOG_FILE, 'a', encoding='utf-8') as file:
        file.write(frase + '\n')

    if (DEBUG): print (frase)

def criar_pasta_simulacao_nova():
    
    if (not(os.path.exists('./logs'))):
        os.mkdir('./logs')
    
    pasta_nova_simulacao = f'./logs/{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")}'
    if (not(os.path.exists(pasta_nova_simulacao))):
        os.mkdir(pasta_nova_simulacao)
    
    return pasta_nova_simulacao

def carregar_checkpoints():
    gen_state_dict = torch.load(f'{PASTA_SIMULACAO}/generator.pt', map_location=torch.device('cpu'))
    disc_state_dict = torch.load(f'{PASTA_SIMULACAO}/discriminator.pt', map_location=torch.device('cpu'))
    resultados = np.load(f'{PASTA_SIMULACAO}/resultados.npy')

    return gen_state_dict, disc_state_dict, resultados

def salvar_checkpoint(_discriminator, _generator, _resultados):
    torch.save(_discriminator.state_dict(), f'{PASTA_SIMULACAO}/discriminator.pt')
    torch.save(_generator.state_dict(), f'{PASTA_SIMULACAO}/generator.pt')
    np.save(f'{PASTA_SIMULACAO}/resultados', np.array(_resultados))
    debug('Modelos salvos!')

def imprimir_resultados(_generator, _dataset, _results, _inv_transformer):
    
    test_dataloader = DataLoader(_dataset, batch_size=len(_dataset))
    real_imgs_batch, target_imgs_batch = next(iter(test_dataloader))
    real_imgs_batch = real_imgs_batch.to(DEVICE)
    target_imgs_batch = target_imgs_batch.to(DEVICE)
    
    _generator.eval()
    with torch.no_grad():
        pred_imgs_batch = _generator(real_imgs_batch)
    
    tensors = torch.vstack([real_imgs_batch, target_imgs_batch, pred_imgs_batch])
    tensors = tensors.detach().cpu()
    imgs_tensor_grid = make_grid(tensors, nrow=4, padding=3)
    imgs_pil = _inv_transformer(imgs_tensor_grid)
    
    imgs_pil.save(f'{PASTA_SIMULACAO}/images.png')

    # salvando a evolução das imagens num vetor numpy para depois fazer uma análise dessa evolução

    imgs_numpy = np.array(imgs_pil)
    if not(os.path.exists(f'{PASTA_SIMULACAO}/evolution.npy')):
        np.save(f'{PASTA_SIMULACAO}/evolution', [imgs_numpy])
    else:
        evolution_array = np.load(f'{PASTA_SIMULACAO}/evolution.npy')
        evolution_array = np.concatenate([evolution_array, [imgs_numpy]])
        np.save(f'{PASTA_SIMULACAO}/evolution', evolution_array)

def salvar_variaveis_log():
    with open(LOG_FILE, 'w') as file:
        file.write(str(args) + '\n')

parser = ArgumentParser()
parser.add_argument('--e', type=int, default=200, help='Numero de epocas no treinamento')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--bs', type=int, default=64, help='Batch Size')
parser.add_argument('--lp', type=str, default='./log', help='Log Path para salvar as imagens geradas a cada epoca.')
parser.add_argument('--d', type=int, default=1, help='Debug do codigo para analisar treinamento.')
parser.add_argument('--new', type=int, default=1, help='Simulacao nova? sim->1, nao->0')
parser.add_argument('--ps', type=str, default='', help='Aproveitar um treinamento e dar continuidade.')
parser.add_argument('--imgs', type=str, default='./content/maps', help='Diretorio onde estao as imagens.')
parser.add_argument('--l1', type=int, default=100, help='L1-LAMBDA. Fator multiplicativo no treinamento do generator.')
parser.add_argument('--imgsize', type=int, default=128, help='IMG_SIZE. Tamanho das imagens para redimensionar.')
parser.add_argument('--gamma', type=float, default=0.6, help='Fator Multiplicativo do Learning Rate a cada sequencia de epocas.')

args = parser.parse_args()
EPOCHS = args.e
LEARNING_RATE = args.lr
BATCH_SIZE = args.bs
LOG_PATH = args.lp
DEBUG = True if args.d == 1 else False
SIMULACAO_NOVA = True if args.new == 1 else False
PASTA_SIMULACAO = args.ps
IMGS_DIR = args.imgs
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
L1_LAMBDA = args.l1
IMG_SIZE = args.imgsize
GAMMA = args.gamma

if (SIMULACAO_NOVA):
    PASTA_SIMULACAO = criar_pasta_simulacao_nova()
else:
    assert os.path.exists(PASTA_SIMULACAO), f'Erro. A pasta {PASTA_SIMULACAO} não existe.'

assert os.path.exists(IMGS_DIR), f'Diretorio: {IMGS_DIR} invalido!!!!!'

LOG_FILE = f'{PASTA_SIMULACAO}/LOG.txt'
salvar_variaveis_log()
print (f'log file: {LOG_FILE}')

imgs_list = [str(l) for l in pathlib.Path(IMGS_DIR).glob('*/*.jpg')]
shuffle(imgs_list)
train_imgs_list = imgs_list[4:]
test_imgs_list = imgs_list[:4]

train_dataset = Dataset_Maps(train_imgs_list, size=IMG_SIZE, tipo='train')
test_dataset = Dataset_Maps(test_imgs_list, size=IMG_SIZE, tipo='test')

images_transformers = Image_Transformer(size=IMG_SIZE)
dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
debug(f'numero de imagens: {len(train_dataset)}')
debug(f'numero de batches: {len(dataloader)}')

generator = Generator(in_channels=3, features=64)
discriminator = Discriminator(in_channels=3, features=[128, 256, 512])
resultados = []
debug('Criacao dos modelos generator, discriminator.')

# INSERIR AQUI UM CARREGAMENTO DE ALGUMA SIMULAÇÃO ANTERIOR.
if not(SIMULACAO_NOVA):
    generator_state_dict, discriminator_state_dict, resultados = carregar_checkpoints()
    debug (generator.load_state_dict(generator_state_dict))
    debug (discriminator.load_state_dict(discriminator_state_dict))
    debug('Aproveitando um treinamento existente e carregando os pesos com sucesso!')

generator.to(DEVICE)
discriminator.to(DEVICE)
debug(f'Modelos jogados para o device: {DEVICE}')

optimizer_generator = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, .999))
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, .999))

MILESTONES = [30, 60, 90]
generator_Scheduler = MultiStepLR(optimizer_generator, milestones=MILESTONES, gamma=GAMMA)
discriminator_Scheduler = MultiStepLR(optimizer_discriminator, milestones=MILESTONES, gamma=GAMMA)

debug('Criando funcoes perdas.')
BCE = nn.BCEWithLogitsLoss().to(DEVICE)
L1_LOSS = nn.L1Loss().to(DEVICE)

debug('Iniciando treinamento.')

for epoch in range(EPOCHS):

    discriminator.train()
    generator.train()
    print (f'epoch: {epoch}, lr: {generator_Scheduler.get_lr()}')

    discriminator_real_loss, discriminator_fake_loss, generator_loss = 0, 0, 0

    for x, y in tqdm(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Treinando o discriminator #################################
        discriminator.zero_grad()

        y_fake = generator(x)
        d_real = discriminator(x, y)
        d_real_loss = BCE(d_real, torch.ones_like(d_real))

        d_fake = discriminator(x, y_fake.detach())
        d_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))

        d_loss = (d_real_loss + d_fake_loss)/2

        d_loss.backward()
        optimizer_discriminator.step()

        # Treinando o generator #####################################
        generator.zero_grad()

        d_fake = discriminator(x, y_fake)
        g_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
        l1 = L1_LOSS(y_fake, y) * L1_LAMBDA
        g_loss = g_fake_loss + l1

        g_loss.backward()
        optimizer_generator.step()

        discriminator_fake_loss += d_fake_loss.item()
        discriminator_real_loss += d_real_loss.item()
        generator_loss += g_loss.item()

    vetor = [time(), discriminator_fake_loss, discriminator_real_loss, generator_loss]
    resultados = np.vstack([resultados, vetor]) if (len(resultados)>0) else np.array([vetor])

    generator_Scheduler.step()
    discriminator_Scheduler.step()

    salvar_checkpoint(discriminator, generator, resultados)
    imprimir_resultados(generator, test_dataset, resultados, images_transformers['inv'])