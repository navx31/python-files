#dictionary in python 
dict1={
    "name":"john",
    "age":"30",
    "city":"new york",
    "job":"developer",
    "salary":"200000"
}
print(dict1)

print(dict1["name"])#accessing value using key
print(dict1.get("age"))#accessing value using get() method
print(dict1.keys())#accessing all keys
print(dict1.values())#accessing all values
print(dict1.items())#accessing all key-value pairs
dict1["age"]=31#updating value
print(dict1)
dict1["email"]="john3456@gmail.com"#adding new key-value pair
print(dict1)
print(dict1.pop("job"))#removing key-value pair using pop() method
print(dict1)
print(len(dict1))#length of dictionary
print(dict1.copy())#copying dictionary
print(dict1.clear())  #clearing dictionary  
print(dict1)#printing empty dictionary after clearing
print(dict1.setdefault("country","USA"))#setting default value for a key
print(dict1)
print(dict1.update({"city":"Los Angeles"}))#updating value using update() method
print(dict1)