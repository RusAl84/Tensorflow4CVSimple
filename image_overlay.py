from PIL import Image
from glob import glob
from random import choice

sign = Image.open('./sign_200x200/Znak_rabotat_v_zaschitnykh_ochkakh_200х200..png')

imgs = glob('./test_bg_1/*')
count = 123
for img in imgs:
    bg = Image.open(img)

    x = choice([25, 30, 35, 40, 45])
    y = choice([25, 30, 35, 40, 45])
    bg.paste(sign.convert('RGB'), (x,y), sign)
    bg.save(f'./test_set/pic_00000{count}.jpg')


