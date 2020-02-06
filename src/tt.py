import numpy as np 


a = np.arange(4).reshape(2,2)
t = np.pad(a, 2, 'wrap')
a = np.repeat(a[np.newaxis, ...], 34, axis=0)
print(a.shape)

a = np.pad(a, ((0,0),(2,2),(2,2)), 'wrap')
for i in range(34):
    print(a[i,...].all() == t.all())
