def float_to_dec_list(x, gamma, n, limit):
    if x < 0.0 or x >= 1.0:
        raise ValueError(x)
    if x <= pow(10, -limit):
        return [0]
    if x >= 1 - pow(10, -limit):
        return [1]
    s = ('%.' + str(limit) + 'f') % x
    result = list(map(int, s[2:]))
    while len(result) > 0 and result[-1] == 0:
        result = result[:-1]
    return [0] + result

def gammapp(x, base):
    x[-1] += 1
    position = len(x) - 1
    while x[position] >= base:
        x[position] -= base
        x[position - 1] += 1
        position -= 1
    
    while x[-1] == 0:
        x = x[:-1]    
    return x

def gammamm(x, base):
    x[-1] -= 1
    position = len(x) - 1
    if x[-1] == 0:
        return x[:-1]
    elif x[-1] > 0:
        return x
    while x[position] < 0:
        x[position] += base
        x[position - 1] -= 1
        position -= 1
    return x
    
def mul(x, num, base):
    add = 0
    #x - list, num - int, base - int
    for i in range(1, len(x))[::-1]:
        x[i] *= num
        x[i] += add
        add = 0
        while x[i] >= base:
            add = x[i] // base
            x[i] %= base
            
    while len(x) > 0 and x[-1] == 0:
        x = x[:-1]
    if len(x) == 0:
        return [add]
    else:
        x[0] += add
    return x
    
def dec_list_to_gamma(x, gamma, n, limit):
    if x == [1]:
        return [1]
    result = []
    while len(result) < limit and len(x) > 1:
        x = mul(x, gamma, 10)
        result.append(x[0])
        x[0] = 0
    return [0] + result

def gamma_to_float(x, gamma):
    mul = 1
    for i in range(len(x)):
        x[i] *= mul
        mul /= gamma
    return sum(x)

def beta(r, n):
    return (pow(n, r) - 1) / (n - 1)

def pre_psi(x, gamma, n, limit):
    """in the end returns float - at midtime returns gamma-list"""
    if x == [1]:
        return 1.0
    k = len(x)
    if k <= 2:
        return gamma_to_float(x, gamma)
    elif k > 2 and x[-1] < gamma - 1:
        return pre_psi(x.copy()[:-1], gamma, n, limit) + x[-1] * pow(gamma, -beta(k - 1, n))
    elif k > 2 and x[-1] == gamma - 1:
        return 0.5 * (pre_psi(gammamm(x.copy(), gamma), gamma, n, limit) + pre_psi(gammapp(x.copy(), gamma), gamma, n, limit))

#####################
    
def psi(x, gamma, n, limit=15):
    """x - float [0, 1]"""
    return pre_psi(dec_list_to_gamma(float_to_dec_list(x, gamma, n, limit), gamma, n, limit), gamma, n, limit)

#####################

def lbd(p, gamma, n):
    """calculates lambda"""
    if p == 1:
        return 1
    r = 1
    lbd_new = 0
    lbd_old = 1
    while lbd_old != lbd_new:
        lbd_old = lbd_new
        lbd_new += pow(gamma, -(p - 1) * beta(r))
        r += 1
    return lbd_new

def inner_function(x, gamma, n, limit=15):
    """x - array of n floats, in1=gamma, in2=n"""    
    a = (1 / gamma) / (gamma - 1)
    
    lbd_arr = [lbd(p + 1, gamma, n) for p in range(n)]
    result = [0] * (2*n+1)
    for q in range(2*n+1):
        for p in range(n):
            result[q] += lbd_arr[p] * psi((x[p] + q * a) % 1.0, gamma, n, limit)
    return result