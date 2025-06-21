import cmath

def fft(a):
    n = len(a)
    if n == 1:
        return a

    even = fft(a[0::2])
    odd = fft(a[1::2])

    y = [0] * n
    for k in range(n // 2):
        w_k = cmath.exp(2j * cmath.pi * k / n)
        y[k] = even[k] + w_k * odd[k]
        y[k + n // 2] = even[k] - w_k * odd[k]

    return y