#set in python 
set1={11,22,33,44,55}
print(set1)
set2={33,44,55,66,77}
print(set2)
print(set1.union(set2))#union of two sets
print(set1.intersection(set2))#intersection of two sets
print(set1.difference(set2))#difference of two sets
print(set2.difference(set1))#difference of two sets
print(set1.symmetric_difference(set2))#symmetric difference of two sets
set1.add(66)#adding element to set
print(set1)
set1.remove(22)#removing element from set
print(set1)
set1.discard(100)#removing element using discard() method
print(set1)
set1.pop()#removing arbitrary element using pop() method
print(set1)
print(len(set1))#length of set
print(33 in set1)#checking membership
print(100 not in set1)#checking non-membership
set3=set1.copy()#copying set
print(set3)
set3.clear()#clearing set
print(set3)
print(set1.isdisjoint(set2))#checking if two sets are disjoint
print(set1.issubset(set2))#checking if set1 is subset of set2
print(set2.issuperset(set1))#checking if set2 is superset of set1
set1.update({77,88,99})#updating set with new elements
print(set1)
set2.intersection_update({55,66,77,88})#updating set2 with intersection
print(set2)