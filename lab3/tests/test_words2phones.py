from lab3_proto import words2phones
from prondict import prondict

if __name__ == "__main__":
    wordTrans = ['z', '4', '3']
    # print(prondict['z'])
    phoneTrans = words2phones(wordTrans, prondict)
    print(phoneTrans)
    ans = ['sil', 'z', 'iy', 'r', 'ow', 'sp', 'f', 'ao', 'r', 'sp', 'th', 'r', 'iy', 'sp', 'sil']
    print(ans)

    print('===========================================================')
    phoneTrans = words2phones(wordTrans, prondict, addSilence=False)
    print(phoneTrans)

    phoneTrans = words2phones(wordTrans, prondict, addSilence=False, addShortPause=False)
    print(phoneTrans)
