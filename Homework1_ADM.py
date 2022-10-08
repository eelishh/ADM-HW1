#!/usr/bin/env python
# coding: utf-8

# ## Say "Hello, World!" With Python

# In[ ]:


print("Hello, World!")


# ## Python If-Else
# 
# 

# In[ ]:


#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())

if n % 2 != 0:
      print ("Weird")
else:
        if n >=2 and n<=5:
          print ("Not Weird")
        elif n >=6 and n<= 20:
           print ("Weird") 
        
        elif n > 20:
                print ("Not Weird")
        


# ## Arithmetic Operators

# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
somma = a + b 
differenza = a - b 
prodotto = a*b

print(somma)
print(differenza)
print(prodotto)


# ## Python: Division

# In[ ]:


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    intdiv = a//b
    floatdiv = a/b
    
    print(intdiv)
    print(floatdiv)


# ## Loops

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    for i in range (0,n):
        print (i**2)


# ## Write a Function

# In[ ]:


def is_leap(year):
    leap = False
    
    # Write your logic here
    if year%400==0 :
        leap = True
    elif year%4 == 0 and year%100 != 0:
        leap = True
    return leap


# ## Print Function

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    for i in range (1,n+1):
        print(i, end = "")
    


# ## List Comprehensions
# 
# 

# In[ ]:


if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    
grid = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n]
print(grid)


# ## Find the Runner-Up Score!

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    sort = sorted(set(arr), reverse = True)
    print (sort[1])


# ## Nested Lists
# 
# 

# In[ ]:



if __name__ == '__main__':
    l = []
    l2 = []
    for i in range(int(input())):
        N = input()
        score = float(input())
        l += [[N,score]]
        l2 +=[score]
    b = sorted(list(set(l2)))[1] 
    for a,c in sorted(l):
        if c==b:
            print(a)


# ## Finding the percentage

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    media =(sum(student_marks[query_name])/len(student_marks[query_name]))
    print('%0.2f' % media)


# ## Lists

# In[ ]:


if __name__ == '__main__':
    N = int(input())
    l = []
    
    for i in range (N):
        x = list(input().split())
        if x[0] == "insert":
           l.insert(int(x[1]), int(x[2])) 
        elif x[0] == "print":
            print(l)
        elif x[0] == "remove":
            l.remove(int(x[1]))
        elif x[0] == "append":
            l.append(int(x[1]))
        elif x[0] == "sort":
            l.sort()  
        elif x[0] == "pop":
            l.pop()
        elif x[0] == "reverse":
            l.reverse()                  
                 


# ## Tuples

# In[ ]:


if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t = tuple(integer_list)
    print(hash(t))


# ## sWAP cASE
# 
# 

# In[ ]:


def swap_case(s):
    return s.swapcase()


# ## String Split and Join

# In[ ]:


def split_and_join(line):
    # write your code here
 return("-".join(line.split(" ")))
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


# ## What's Your Name?

# In[ ]:


def print_full_name(first, last):
    print("Hello " + first, last + "! You just delved into python.")


# ## Mutations

# In[ ]:


def mutate_string(string, position, character):
    l=list(string)
    l[position]= character
    string = "".join(l)
    return(string)


# ## Find a string

# In[ ]:


def count_substring(string, sub_string):
    count = 0
    for a in range(len(string)-len(sub_string)+1):
        if (string[a:a + len(sub_string)] == sub_string):
            count += 1
    return count


# ## String Validators

# In[ ]:


s = input()
print(any(map(str.isalnum, s)))
print(any(map(str.isalpha, s)))
print(any(map(str.isdigit, s)))
print(any(map(str.islower, s)))
print(any(map(str.isupper, s)))


# ## Text Alignment

# In[ ]:


#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# ## Text Warp

# In[ ]:


def wrap(string, max_width):
    return textwrap.fill(string,max_width)


# ## Designer Door Mat
# 
# 

# In[ ]:


N, M= map(int, input().split())
for i in range(1, N, 2):
    print(('.|.'*i).center(M,'-'))
print('WELCOME'.center(M,'-'))
for i in range(N-2, -1, -2):
    print(('.|.'*i).center(M, '-'))


# ## String Formatting

# In[ ]:


def print_formatted(number):
    l = len(bin(number)[2:])
    for i in range(1, number+1):
        decimal = str(i)
        octal = oct(i)[2:]
        hexadec = hex(i)[2:].upper()
        binary = bin(i)[2:]

        print(decimal.rjust(l), octal.rjust(l), hexadec.rjust(l),binary.rjust(l))
 


# ## Alphabet Rangoli

# In[ ]:


def print_rangoli(n):
 a = "abcdefghijklmnopqrstuvwxyz"
 for i in range(n-1, -n, -1):
    string = '-'.join(a[n-1:abs(i):-1] + a[abs(i):n])
    print(string.center(4 * n - 3, '-'))


# ## Capitalize!

# In[ ]:


def solve(s):
    r = ""
    for i in range(len(s)):
        if i == 0:
            r += s[i].capitalize()
        elif s[i-1] == " " and s[i] != " ":
             r += s[i].capitalize()
        else:
            r += s[i]         
    return(r)


# ## Merge the tools!
# 

# In[ ]:


def merge_the_tools(string, k):
    for i in range(0,len(string),k):
        string2 = []
        for m in range(i,i+k):
            if string[m] not in string2:
                string2.append(string[m])
        sub_seq=''.join(string2)
        print(sub_seq)
        string2.clear()


# ## Introduction to Sets

# In[ ]:


def average(array):
 somma = sum(set(array))
 tot = len(set(array))
 
 return(somma/tot)


# ## No Idea!

# In[ ]:


N, M = input().split()
arr = input().split()
A = set(input().split())
B = set(input().split())
C = 0
for i in arr:
    if i in A:
     C += 1
    elif i in B:
     C -= 1
print(C)


# ## Symmetric Difference

# In[ ]:


M = int(input())
setM = set(map(int, input().split()))
N = int(input())
setN = set(map(int, input().split()))
difftot = setM.difference(setN).union(setN.difference(setM))

for i in sorted(list(difftot)):
    print (i)


# ## Set.add()

# In[ ]:


N = int(input())
S = set(input() for i in range(N))
print(len(S))


# ## Set .discard(), .remove() & .pop()
# 
# 

# In[ ]:


n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
   snew = input().split()
   if snew[0]=="pop":
    s.pop()
   elif snew[0] == "remove":
    s.remove(int(snew[1]))
   elif snew [0] == "discard":
    s.discard(int(snew[1]))
     
print(sum(s))  


# ## Set .intersection() Operation
# 
# 

# In[ ]:


eng = int(input())
neng = set(input().split())
fr = int(input())
nfr = set(input().split())
print(len(neng.intersection(nfr)))


# ## Set .union() Operation
# 
# 

# In[ ]:


eng = int(input())
neng = set(input().split())
fr = int(input())
nfr = set(input().split())
print(len(neng.union(nfr)))


# ## Set .difference() Operation

# In[ ]:


eng = int(input())
neng = set(input().split())
fr = int(input())
nfr = set(input().split())
print(len(neng.difference(nfr)))


# ## Set .symmetric_difference() Operation

# In[ ]:


eng = int(input())
neng = set(input().split())
fr = int(input())
nfr = set(input().split())
print(len(neng.symmetric_difference(nfr)))


# ## Set mutations

# In[ ]:


A = int(input())
S = set(map(int,input().split()))
N = int(input())
for i in range(N):
    x,y=input().split()
    if x=='intersection_update':
       S.intersection_update(set(map(int,input().split())))
    
    elif x=='update':
        S.update(set(map(int,input().split())))    
    
    elif x=='symmetric_difference_update':
        S.symmetric_difference_update(set(map(int,input().split())))
    
    elif x=='difference_update':
        S.difference_update(set(map(int,input().split())))

print(sum(S))


# ## The Captain's Room
# 
# 

# In[ ]:


K = input ()
S1 = set ()
S2 = set()
for i in (input ().split ()) : 
    if i not in S1: 
        S1.add(i)
    else:
        S2.add(i)
S1.difference_update(S2)
print(S1.pop())


# ## Check Subset
# 
# 

# In[ ]:


for i in range (int(input())):
    a = int(input())
    A = set(map(int, input().split()))
    b = int(input())
    B = set(map(int, input().split()))
    if len(A-B) == 0:
     print('True')
    else:
     print('False')    
    


# ## Check Strict Superset

# In[ ]:


A = set(input().split())
N = int(input())
c = 0
check = True
for _ in range(N):
    if not A.issuperset(set(input().split())):
        check = False
        break
print(check)


# ## collections.Counter()

# In[ ]:


from collections import Counter
X = int(input())
a = list(map(int, input().split()))
costumers = int(input())
totprice = 0
conta = Counter(a)
for i in range(costumers):
    size, price = map(int, input().split())
    if size in conta.keys():
      if conta[size] > 0:
        totprice += price 
        conta[size] -= 1
print(totprice)


# ## DefaultDict Tutorial

# In[ ]:


from collections import defaultdict
D = defaultdict(list)
N,M = map(int, input().split())
for i in range (N):
    D[input()].append(str(i + 1))
for j in range(M):
    print(' '.join(D[input()]) or -1)


# ## Collections.namedtuple()

# In[ ]:


from collections import namedtuple
N = int(input())
col = namedtuple("col", input().split())
summ = 0
for i in range(N):
    r = col(*input().split())
    summ += int(r.MARKS)
print(float(summ)/float(N))    


#  ## Collections.OrderedDict()
# 
# 

# In[ ]:


from collections import OrderedDict
N = int(input())

order = OrderedDict()
for i in range(N):
    elenco = input().split()
    cibo = " ".join(elenco[:-1])
    prezzo = int(elenco[-1])
    if cibo in order:
       order[cibo] += prezzo
    else:
        order[cibo] = prezzo
for u in order:
    print(u, order[u])           


# ## Collections.deque()
# 
# 

# In[ ]:


from collections import deque
N = int(input())
d = deque()

for i in range(N):
    com = input().split()   
    if com[0] == 'append':
        d.append(com[1])
    elif com[0] == 'pop':
        d.pop()
    elif com[0] == 'popleft':
        d.popleft()
    elif com[0] == 'appendleft':
        d.appendleft(com[1])  
print(*d)    


# ## Word Order
# 
# 

# In[ ]:


from collections import Counter

N = int(input())
l = [input() for i in range(N)]

count = Counter(l)
countl = [i for i in count.values()] 

print(len(set(l)))
for i in countl:
    print(i, end=' ')


# ## Company Logo
# 
# 

# In[ ]:


import math
import os
import random
import re
import sys

from collections import Counter

if __name__ == '__main__':
 s = input()
 count = Counter(sorted(s))

for i in count.most_common(3):
    print(*i) 


# ## Calendar Module

# In[ ]:


import calendar
import datetime
M, D, Y = map(int, input().split())
date = datetime.date (Y, M, D)
print(calendar.day_name[date.weekday()].upper())


# ## Time delta

# In[ ]:


import math
import os
import random
import re
import sys

# Complete the time_delta function below.
from datetime import datetime
def time_delta(t1, t2):
    format_data = "%a %d %b %Y %H:%M:%S %z"
    a = datetime.strptime(t1,format_data)
    b = datetime.strptime(t2,format_data)
    out = int(abs(a-b).total_seconds())
    return str(out)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()


# ## Exceptions
# 
# 

# In[ ]:


T = int(input())
for x in range(T):
    try:
        a, b = input().split()
        print (int(a)//int(b))
    except ZeroDivisionError as e:
        print ("Error Code:",e)
        
    except  ValueError as VE: 
        print ("Error Code:",VE)


# ## Zipped!
# 
# 

# In[ ]:


N, X = input().split()
sub= []
for i in range(int(X)):
    ip = map(float, input().split())
    sub.append(ip)
for a in zip(*sub): 
    print(sum(a)/len(a))


# ## Athlete Sort
# 
# 

# In[ ]:


import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []
for i in range(n):
    arr.append(list(map(int,input().split())))
k = int(input())
arr.sort(key=lambda arr:arr[k])
for j in arr:
    print(*j)


# ## ginortS
# 
# 

# In[ ]:


import re
s = input()
low = re.findall('[a-z]',s)
up = re.findall('[A-Z]',s)

odd_dig = re.findall(r'[1,3,5,7,9]',s)
even_dig = re.findall(r'[0,2,4,6,8]',s) 
    
print("".join (sorted(low))+"".join(sorted(up))+"".join(sorted(odd_dig))+"".join(sorted(even_dig)))  


# ## Map and Lambda Function

# In[ ]:


cube = lambda x: pow(x,3)
def fibonacci(n):
    lista = []
    a, b = 0, 1
    for _ in range(n):
        lista.append(a)
        a, b = b, a + b
    return lista


# ## Detect Floating Point Number

# In[ ]:


import re
T = int(input())
for i in range(T):
 a = input()
 print(bool(re.match("^[-|+]?\d*[.]\d+$", a)))


# ## Re.split()
# 
# 

# In[ ]:


regex_pattern = r"[,.]"	# Do not delete 'r'.


# ## Group(), Groups() & Groupdict()
# 
# 

# In[ ]:


import re
s = re.search(r'([a-zA-Z0-9])\1+', input())
print(s.group(1) if s else -1)


# ## Re.start() & Re.end()

# In[ ]:


import re
s = input()
k = input()
if(bool(re.findall(k,s))):
    
 for i in range(len(s)):
    if(bool(re.match(k,s[i:]))):
        print((i,i+len(k)-1))
else:
 print((-1,-1)) 


# ## Re.findall() & Re.finditer()
# 
# 

# In[ ]:


import re
f = re.finditer(r"(?<=[^aeiouAEIOU])([aeiouAEIOU]){2,}(?=[^aeiouAEIOU])", input())
l = [*f]
if len(l) > 0:
    for i in l:
        print(i.group())
else:
    print(-1)


# ## Validating Roman Numerals

# In[ ]:


regex_pattern = r"M{0,3}(CD|D?C{0,3}|CM)(XL|L?X{0,3}|XC)(IV|V?I{0,3}|IX)$"


# ## Validating phone numbers

# In[ ]:


import re
N = int(input())
for i in range(N):
    if bool(re.search(r'^[7-9]\d{9}$', input())) == True:
        print('YES')
    else:
        print('NO')


# ## Hex Color Code
# 
# 

# In[ ]:


import re
N = int(input())
c = re.compile("(?<!^)(#(?:[\dA-Fa-f]{3,6}))")
for i in range(N) :
    l= input()
    f = c.findall(l)
    if f:
        print(*f, sep='\n')


# ## XML 1 - Find the Score
# 
# 

# In[ ]:


def get_attr_number(node):
 return sum([len(elem.items()) for elem in node.iter()])


# ## XML2 - Find the Maximum Depth

# In[ ]:


maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    maxdepth = max(maxdepth, level)
    for i in elem:
        depth(i,level)
    return maxdepth 


# ## Standardize Mobile Number Using Decorators
# 
# 

# In[ ]:


def wrapper(f):
    def fun(l):
     lista =['+91 '+i[-10:-5]+' '+i[-5:] for i in l]
     f(lista)
    return fun


# ## Arrays
# 
# 

# In[ ]:


def arrays(arr):
   arr.reverse()
   a = numpy.array(arr, float)
   return(a)


# ## Shape and Reshape

# In[ ]:


import numpy
N = numpy.array(input().split(' '), int)
N.shape = (3,3)

print(N)


# ## Transpose and Flatten
# 
# 

# In[ ]:


import numpy
N, M = map(int, input().split())
arr= numpy.array([list(map(int,input().split())) for i in range(N)])
print(numpy.transpose(arr))
print(arr.flatten())


# ## Sum and Prod

# In[ ]:


import numpy
N, M = map(int, input().split())
arr = numpy.array([input().split() for _ in range(N)], int)
print(numpy.prod(numpy.sum(arr, axis=0)))


# ## Floor, Ceil and Rint
# 
# 

# In[ ]:


import numpy
numpy.set_printoptions(sign=' ')
A = numpy.array(input().split(),float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))


# ## Array Mathematics
# 
# 

# In[ ]:


import numpy
N, M = map(int, input().split())

A = numpy.array([list(map(int, input().split())) for i in range(N)])
B = numpy.array([list(map(int, input().split())) for i in range(N)])

print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)


# ## Eye and Identity
# 
# 

# In[ ]:


import numpy
numpy.set_printoptions(sign=' ')
print(numpy.eye(*map(int, input().split())))


# ## Zeros and Ones
# 
# 

# In[ ]:


import numpy
N = tuple(map(int, input().split()))
print(numpy.zeros(N, int))
print(numpy.ones(N, int))


# ## Concatenate
# 
# 

# In[ ]:


import numpy as np
m, n, p = map(int,input().split())
arr1 = np.array([input().split() for _ in range(m)],int)
arr2 = np.array([input().split() for _ in range(n)],int)
print(np.concatenate((arr1, arr2), axis = 0))


# ## Dot and Cross
# 
# 

# In[ ]:


import numpy
N = int(input())
A = numpy.array([input().split() for i in range(N)], int)
B = numpy.array([input().split() for i in range(N)], int)
print(numpy.dot(A,B))


# ## Min and Max

# In[ ]:


import numpy
N , M = map(int, input().split())
arr=numpy.array([list(map(int,input().split())) for i in range(N)])
print(max(numpy.min(arr, axis = 1)))


# ## Inner and Outer
# 
# 

# In[ ]:


import numpy
A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)
print(numpy.inner(A, B))
print(numpy.outer(A, B))


# ## Polynomials

# In[ ]:


import numpy
P = map(float, input().split())
x = int(input())
print(numpy.polyval(list(P),x))


# ## Linear Algebra
# 
# 

# In[ ]:


import numpy
N = int(input())
A = numpy.array([input().split() for i in range(N)],float)
print(round(numpy.linalg.det(A),2))


# ## Birthday Cake Candles
# 
# 

# In[ ]:


import math
import os
import random
import re
import sys


def birthdayCakeCandles(candles):
    # Write your code here
    count = 0   
    max = 0
    for i in candles:
        if i > max:
            max = i
            count = 1 
        elif i == max:
            count += 1
    return(count)        
             
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


# ## Number Line Jumps

# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    jump = x2-x1
    for i in range(jump):
        x1 = x1 + v1
        x2 = x2 + v2
        if x1 == x2:
            return("YES")
    return("NO")
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# ## Viral Advertising
# 
# 

# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    # Write your code here

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# ## Recursive Digit Sum
# 
# 

# In[ ]:


import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    s = sum(list(map(int,str(n))))
    m = s*k
    def function(m):
        if len(str(m)) == 1:
            return m
        else:
             c = sum(list(map(int,str(m))))  
             m = c 
        return function(m)
    f = function(m)
    return f
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# ## Insertion Sort - Part 1

# In[ ]:



import math
import os
import random
import re
import sys


def insertionSort1(n, arr):
 a = arr[-1]
 for i in range(len(arr)-2,-1,-1):
        if arr[i] < a:
            arr[i+1], a = a ,"X"
            break
        else:
            arr[i+1]=arr[i]
            print(*arr)
 if a!="X": arr[0]= a
 print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# ## Insertion Sort - Part 2

# In[ ]:


import math
import os
import random
import re
import sys


def insertionSort2(n, arr):
   for i in range(1, n):
    p = 0
    for j in range(0, i+1):
        if arr[i] > arr[j]:
            p += 1
        else:
            break
    arr.insert(p, arr.pop(i))
    print(*arr)
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)

