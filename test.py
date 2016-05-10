import hmm
import string, time
import numpy as np
import pylab as pl


if __name__ == '__main__':
  corpus = ['corpus.low']

  alpha = string.ascii_lowercase
  space = ' '

  vocab = alpha + space
  dic = dict(zip(vocab, range(len(vocab))))
  inv_dic = {v:k for k,v in dic.items()}

  O = []
  for f in corpus:
    with open(f, 'r') as fp:
      s = fp.read().strip()
    O.append(np.asarray([dic[_] for _ in s if _ in dic], dtype='int32'))

  n_A = 2
  n_B = len(dic)

  A = np.random.rand(n_A,n_A) + 3
  A /= A.sum(axis=1)[:,np.newaxis]
  B = np.random.rand(n_A,n_B) + 3
  B /= B.sum(axis=1)[:,np.newaxis]
  start = np.random.rand(n_A)
  start /= start.sum()

  np.set_printoptions(precision=5, suppress=True)
############################################################

  begin = time.time()
  print('begin =', begin)
  hmm.baum_welch(A, B, start, O, 5000, eps=1e-10, verbose=False)
  print('cost =', time.time() - begin)

  print(A)
  print(start)

  print('emission prob:')  
  for i in range(n_B):
    print(inv_dic[i], B[:,i])

  pl.style.use('ggplot')
  fig, ax = pl.subplots(2,1)
  pl.setp(ax, xticks=range(len(vocab)), 
      xticklabels=vocab,
      xlim=(-1, len(vocab)),
      ylim=(0,1))
  ax[0].stem(range(len(vocab)), B[0], markerfmt=' ')
  ax[1].stem(range(len(vocab)), B[1], markerfmt=' ')
  pl.show()
