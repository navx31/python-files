list1 = [43, 54, 65, 76]
list2 = [32, 21, 12, 23]
list3=list1+list2
print(list3)
print(list3*3)
print(54 in list3)
print(54 not in list3)
print(len(list3))
print(max(list3))
print(min(list3))
print(sum(list3))
print(sorted(list3))
print(list(reversed(list3)))

print(list3.index(54))

list3.append(100)
print(list3)

list3.insert(2,200)
print(list3)

list3.remove(65)
print(list3)

list3.pop()
print(list3)

list3.pop(2)
print(list3)

list3[2]=300
print(list3)

list3.sort()
print(list3)

list3.clear()
print(list3)

