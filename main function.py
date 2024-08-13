def mainf():
    dir='./image\\'
    LIst=get_filelist(dir, [])
    print(LIst)
    ST=0
    for ii in range(len(LIst)-1):
        name=LIst[ii][8:]
        print(name)
        namep=name[:-4]
        print(namep)
        time=run(name,namep)
        print(time)
        ST=ST+time
    print(ST/300)


if __name__ == '__main__':
    mainf()
