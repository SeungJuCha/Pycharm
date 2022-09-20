def fib1(n): # return Fibonacci series up to n
    """Return a list containing the Fibonacci series up to n."""
    # """"""로 이 함수를 설명하는 내용
    result =[]
    a,b =0,1 #더블 대입
    while a <n:
        print(a,end ="  ") # see below
        a,b =b,a+b
    print()

def fib2(n): # return Fibonacci series up to n
    """Return a list containing the Fibonacci series up to n."""
    # """"""로 이 함수를 설명하는 내용
    result =[]
    a,b =0,1 #더블 대입
    while a <n:
        result.append(a) # see below
        a,b =b,a+b
    return result