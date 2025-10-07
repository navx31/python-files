list1 = [32, 21, 12, 23, 43, 34]
list1.sort()
total = sum(list1)
multiplication = 1
for num in list1:
    multiplication *= num

print("The sorted list is:", list1)
print("The sum of the numbers is:", total)
print("The multiplication of the numbers is:", multiplication)

