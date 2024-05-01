#functions
def greet():
    print("Hello world:)")


greet()

#functions with parameters

def add(n1,n2):
    print(n1+n2)

add(2,5)


#integers are immutable, thus even if the value changes during the function call. It remains the same before and after the function execution
def incr(n):
    n+=1
    print("during fn: ",n)

n=10
print("before function: ", n)
incr(n)
print("after function: ", n)

#on similar lines, lists are mutable

def lesgo(lst):
    lst.append(23)
    print("during: ", lst)
lst=[1,2]
print("before function: ", lst)
lesgo(lst)
print("after function: ", lst)


#there also exists return statement where the result is not directly printed, but is returned to the user behind the screen you can print it or do required operations on it
def add(n1,n2):
    return(n1+n2)
print(add(5,7))

#types of arguments
#default argument

def add(n1,n2=60):
    print(n1+n2)

add(10)

#but if we do
add(50,70)

#arbitrary arguments

def sums(n1, *args):
    s=0
    for i in args:
        s+=i
    print(s)

sums(1,2,3,4)


#lambda function - a random function without actually defining it.

