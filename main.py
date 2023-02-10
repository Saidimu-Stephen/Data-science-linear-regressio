import numpy as np
# num1 =int( input('Enter the first number: '))
# num2 = int(input('Enter the second number: '))
# sum= num1+num2
# print('The sum of {0} and {1} is {2}: '.format(num1, num2, sum))
#
#
# print("Enter you score in each subject")
# sub1=int(input('First subject: '))
# sub2=int(input('Second subject: '))
# sub3=int(input('Third subject: '))
# sub4=int(input('Fourth subject: '))
# total=sub1+sub2+sub3+sub4
# avg= total/4
# if avg>=70:
#     grade='A'
# elif avg>=60:
#     grade='B'
# elif avg >=50:
#     grade='C'
# elif avg >=40:
#     grade='D'
# else:
#     grade='Fail'
#
# print('The average marks score is {0} and the grade is {1}'.format(avg,grade))
#
# def print_numbers():
#     for num in range(1, 21):
#         print(num)
#
# print_numbers()


# def print_numbers():
#     number = 1
#     while number <= 20:
#         print(number)
#         number += 1
#
# def sum_numbers():
#     sum = 0
#     for i in range(1, 21):
#         sum += i
#     print(sum)
#
# print_numbers()
# sum_numbers()

#

# arr= np.array([23,45,5,52,56])
# print (arr)
# print(arr.shape)
# print(arr.size)

print('****multi dimention array******')
arr= np.array([[23,45,5,52,56], [34,56,77,22,12]])
print (arr)

a=np.ones((10,10), dtype=int)
print (a)

for i in range(1,11):
    print(i)

print('.................range.................')

numbers=[1,2,3,4,5]
for i in numbers:
    print(i)

print("identity")
arr= np.identity(10)
print(arr)

print("identity")
arr= np.eye(5, dtype=int)
print(arr)


print('random')
arr = np.random.randn(3, 4)
print(arr)


print('uniform distribution')
np.random.random((3, 3))
print(arr)

