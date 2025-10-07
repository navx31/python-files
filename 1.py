#hello world program
print("Hello, World!")

#taking input from user
name=input("Enter your name:")
age=int(input("Enter your age:"))
print("Welcome ",name)
print("your age is",age)

#area of retangle
lenght=int(input("Enter the lenght of retangle:"))
breadth=int(input("Enter the breadth of retangle:"))
area=lenght*breadth
print("The area of retangle is:",area)

#swapping of two numbers
Num1=23
Num2=45
print("before swapping: ",Num1,Num2)
Num1,Num2=Num2,Num1
print("after swapping: ",Num1,Num2)

#check whether the number is even or odd
a=int(input("Enter a number:"))
if(a%2==0):
    print("The number is even")
else:
    print("The number is odd")