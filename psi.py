def float_to_dec_list(x):
    if x < 0.0 or x >= 1.0:
        raise ValueError(x)
    if x == 0.0:
        return [0]
    s = ('%.' + str(limit) + 'f') % x
    result = list(map(int, s[2:]))
    while result[-1] == 0:
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
    
def division(x, num, base):
    add = 0
    #x - list, num - int, base - int
    for i in range(limit):
        if add > 0 and i >= len(x):
            x.append(0)
        elif add == 0 and i >= len(x):
            return x
        x[i] += add * base
        add = x[i] % num
        x[i] = x[i] // num        
    return x

def dec_list_to_gamma(x):
    result = []
    while len(result) < limit and len(x) > 1:
        x = mul(x, gamma, 10)
        result.append(x[0])
        x[0] = 0
    return [0] + result

def gamma_to_float(x):
    mul = 1
    for i in range(len(x)):
        x[i] *= mul
        mul /= gamma
    return sum(x)

def beta(r):
    return (pow(n, r) - 1) / (n - 1)
def pre_psi(x):
    """in the end returns float - at midtime returns 
gamma-list"""
    k = len(x)
    if k <= 2:
        return gamma_to_float(x)
    elif k > 2 and x[-1] < gamma - 1:
        return pre_psi(x.copy()[:-1]) + x[-1] * pow(gamma, -beta(k - 1))
    elif k > 2 and x[-1] == gamma - 1:
        return 0.5 * (pre_psi(gammamm(x.copy(), gamma)) + pre_psi(gammapp(x.copy(), gamma)))

#####################
    
def psi(x, in1, in2, lim=15):
    """x - float [0, 1]"""
    global gamma
    global n
    global limit
    gamma = in1
    n = in2
    limit = lim
    return pre_psi(dec_list_to_gamma(float_to_dec_list(x)))

#####################

def lbd(p, gamma, in2):
    """calculates lambda"""
    global n
    n = in2
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

def inner_function(x, in1, in2, lim=15):
    """x - array of n floats, in1=gamma, in2=n"""
    global gamma
    global n
    global limit
    gamma = in1
    n = in2
    limit = lim
    
    a = (1 / gamma) / (gamma - 1)
    
    lbd_arr = [lbd(p + 1, gamma, n) for p in range(n)]
    result = [0] * (2*n+1)
    for q in range(2*n+1):
        for p in range(n):
            result[q] += lbd_arr[p] * psi((x[p] + q * a) % 1.0, gamma, n, limit)
    return result