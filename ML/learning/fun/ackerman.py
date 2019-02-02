#Robert Nool III
#Ackermann's Function


#Start

def ackermann(m,n):
     if m == 0:
          return (n + 1)
     elif n == 0:
          return ackermann(m - 1, 1)
     else:
          return ackermann(m - 1, ackermann(m, n - 1)) 
          
x=int(input("What is the value for m? "))
print(x)

y=int(input("What is the value for n? "))
print(x)

print("\nThe result of your inputs according to the Ackermann Function is:")
print(ackermann(x, y)) 

#End


