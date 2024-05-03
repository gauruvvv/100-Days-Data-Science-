

# making even and odd list

even= [i for i in range(1,21) if i%2==0]
odd = [i for i in range(1,21) if i%2!=0]
odd.remove(1)
print(even)
print(odd)

#making a matrix

mat = [[i for i in range (3)]for j in range(3)]
print(mat)

#letters of a word into string

letters = [i for i in 'Gaurav']
print(letters)

#Tuples
#basic ops

tuple = (1,2,3,4,5)
print(len(tuple))
print(type(tuple))
print(min(tuple))
print(max(tuple))
print(sum(tuple))

#tuple is indexed 0 to n-1 and -1 to -n

t = (1,2,3,4,5,6)
print(t[2])
print(t[-1])
print(t[-6])

#tuple indexing

p = (1,2,3,4,2,3,4,5,2,3,4,2,3,1,3,2)
print(p)
print(p.index(2))
#p.index(2, 2, -1) searches for the value 2 starting from index 2 up to index -1
print(p.index(2,2,-1))
