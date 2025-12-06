class student:
    def __init__(self,Name,Marks):
        self.Name=Name
        self.Marks=Marks

    def display(self):
        print(self.Name,self.Marks)

s1=student("raj",85)    
s1.display()

#encapsulation
class bank_account:
    def __init__(self,account_number,balance):
        self.__account_number=account_number
        self.__balance=balance

    def deposit(self,amount):
        self.__balance+=amount
        print(f"Deposited {amount}. New balance is {self.__balance}")

    def withdraw(self,amount):
        if amount > self.__balance:
            print("Insufficient funds")
        else:
            self.__balance-=amount
            print(f"Withdrew {amount}. New balance is {self.__balance}")

    def get_balance(self):
        return self.__balance
account=bank_account("123456789",1000)
account.deposit(500)
account.withdraw(200)
print(f"Current balance is {account.get_balance()}")

#inheritance
class Animal:
    def sound(self):
        print("Animal makes a sound")
class Dog(Animal):
   pass
d=Dog()
d.sound()

#polymorphism
class Cat(Animal):
    def sound(self):
        print("Cat meows")
c=Cat()
c.sound()

def add(a,b):
    return a+b  
def add(a,b,c):
    return a+b+c    
print(add(2,3))
print(add(2,3,4))


