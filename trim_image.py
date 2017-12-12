from os import listdir, rename
from os.path import isfile, join
from skimage.io import imread

def main():
    onlyfiles = [f for f in listdir("train2017/train2017") if isfile(join("train2017/train2017", f))]
    print("image: {}".format(onlyfiles[0]))
    
    for f in onlyfiles:
        img = imread(join("train2017/train2017", f))
        if img.shape[0] < 224 or img.shape[1] < 224:
            rename(join("train2017/train2017", f), join("train2017/undersized", f))
    
    
if __name__ == '__main__':
    main()