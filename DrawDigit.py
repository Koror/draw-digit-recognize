import pygame
import numpy as np
from PIL import Image
from NNRecognizer import NNRecognizer


NN = NNRecognizer()
pygame.init()
sc = pygame.display.set_mode((280, 280))
clock = pygame.time.Clock()

FPS = 60

sc.fill((0, 0, 0))
pygame.display.update()
n_img = np.array(())
while True:
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            exit()
        if i.type == pygame.KEYDOWN:
            if i.key == pygame.K_r:
                n_img = n_img.reshape((28,28,1))
                n_img = n_img.reshape((1,28,28,1))
                print(NN.predict(n_img))
            if i.key == pygame.K_c:
                sc.fill((0, 0, 0))
                pygame.display.update()
        pressed = pygame.mouse.get_pressed()
        pos = pygame.mouse.get_pos()
        if pressed[0]:
            pygame.draw.circle(sc, (255,255,255), pos, 10)
            pygame.display.update()
            screensurf = pygame.display.get_surface()
            screenarray = pygame.surfarray.pixels2d(screensurf)
            screenarray = np.where(screenarray == 16777215, 1, screenarray)
            img = Image.fromarray(screenarray)
            img.thumbnail((28,28))
            n_img = np.asarray(img)
            n_img = np.swapaxes(n_img,1,0)


    pygame.time.delay(2)
